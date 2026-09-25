"""Splitting shell command lines into their chained commands.

The execution allowlist wants to match chained commands on one line — e.g.
`cd /x && ls` should satisfy an `ls` rule — and the redundant-cd stripper
shares the same chain-operator semantics. This is the shared, shell-agnostic
splitter both use, so the rules never drift apart.
"""

import re


def split_shell_chain(line, operators=("&&", "||", ";", "&")):
    """Split a shell line into its top-level chained commands.

    Splits on ``operators`` (by default ``&&``, ``||``, ``;``, ``&``) only
    when they sit at the top level — outside quotes and not escaped — so
    ``cd "a && b"`` stays one command and ``cd /x\\ with\\ space && ls``
    splits on the ``&&`` but not the escaped spaces. A bare ``|`` is NOT a
    chain operator here: a pipeline is treated as a single command. Empty
    pieces (e.g. the dangling part of ``cd /x &&``) are dropped.
    """
    pieces = []
    current = []
    quote = None
    i = 0
    n = len(line)
    ops = sorted(operators, key=len, reverse=True)
    while i < n:
        ch = line[i]
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            current.append(ch)
            current.append(line[i + 1])
            i += 2
            continue
        op = next((o for o in ops if line.startswith(o, i)), None)
        if op:
            pieces.append("".join(current))
            current = []
            i += len(op)
            continue
        current.append(ch)
        i += 1
    pieces.append("".join(current))
    return [piece.strip() for piece in pieces if piece.strip()]


_REDIRECT_OPERATORS = {
    "posix": re.compile(r"(?:[0-9]?>{1,2}\|?|&>{1,2}\|?)"),
    "powershell": re.compile(r"(?:[1-6*]?>{1,2})"),
}
_COMMENT_BOUNDARY_CHARS = " \t\r\n;&|(){}=,"
_BASH_HEREDOC_RE = re.compile(
    r"<<-?[ \t]*(?:\"([^\n\"]*)\"|'([^'\n]*)'|([^\s;&|(){}<>]+))"
)
_NULL_TARGET_END_CHARS = " \t\r\n;&|(){}<>"


def replace_null_redirect_target(code, null_device, powershell=False):
    """Replace a bare shell redirect target named `nul` with the real null sink.

    Bash and PowerShell have no filename spelled plain ``nul`` that behaves as
    a device: ``command 2>nul`` ordinarily creates a file literally called
    ``nul``. On Windows that name can collide with the reserved NUL device and
    then be difficult to remove. Use ``/dev/null`` for POSIX shells and
    ``$null`` for PowerShell. Quoted strings, comments, here-documents, other
    ``nul``-prefixed names, and redirect sources are left alone.

    Returns ``(new_code, changed)``.
    """
    if null_device is None:
        return code, False

    style = "powershell" if powershell else "posix"
    operator_re = _REDIRECT_OPERATORS[style]
    output = []
    changed = False
    i = 0
    n = len(code)
    quote = None

    def is_boundary(index):
        if index <= 0:
            return True
        return code[index - 1] in _COMMENT_BOUNDARY_CHARS

    def find_quoted_end(start):
        """Return the end offset just after a valid quoted `nul` target."""
        nonlocal quote
        opening = code[start]
        quote = opening
        j = start + 1
        while j < n:
            char = code[j]
            if not powershell:
                if char == "\\" and j + 1 < n:
                    j += 2
                    continue
            elif char == "`" and j + 1 < n:
                j += 2
                continue
            elif char == opening and j + 1 < n and code[j + 1] == opening:
                # PowerShell escapes a quote by doubling it.
                j += 2
                continue
            if char == opening:
                quote = None
                return j + 1
            j += 1
        # Unterminated quotes are unsafe to parse; leave the rest as-is.
        quote = None
        return None

    while i < n:
        char = code[i]
        if quote is None:
            if not powershell:
                heredoc_end = _bash_heredoc_end(code, i)
                if heredoc_end is not None:
                    output.append(code[i:heredoc_end])
                    i = heredoc_end
                    continue
            elif _powershell_here_string_end(code, i) is not None:
                end = _powershell_here_string_end(code, i)
                output.append(code[i:end])
                i = end
                continue
            if char in "'\"":
                quote = char
                output.append(char)
                i += 1
                continue
            if char == "\\" and i + 1 < n:
                # Preserve escapes such as `\>` or `\"`; they are not operators.
                output.append(code[i : i + 2])
                i += 2
                continue
            if powershell and code.startswith("<#", i):
                end = _powershell_block_comment_end(code, i)
                output.append(code[i:end])
                i = end
                continue
            if char == "#" and is_boundary(i):
                end = code.find("\n", i)
                if end == -1:
                    output.append(code[i:])
                    break
                output.append(code[i:end])
                i = end
                continue
            operator = operator_re.match(code, i)
            if operator:
                output.append(operator.group(0))
                i = operator.end()
                while i < n and code[i] in " \t\r\n":
                    output.append(code[i])
                    i += 1
                if i < n and code[i] in "'\"" and code[i - 1] != "$":
                    end = find_quoted_end(i)
                    if (
                        end is not None
                        and code[i + 1 : end - 1].lower() == "nul"
                        and (
                            end == n
                            or code[end] in _NULL_TARGET_END_CHARS
                        )
                    ):
                        output.append(null_device)
                        changed = True
                        i = end
                        continue
                    if end is None:
                        output.append(code[i:])
                        break
                    output.append(code[i:end])
                    i = end
                    continue
                if (
                    code[i : i + 3].lower() == "nul"
                    and (i + 3 == n or code[i + 3] in _NULL_TARGET_END_CHARS)
                ):
                    output.append(null_device)
                    changed = True
                    i += 3
                    continue
                continue
            output.append(char)
            i += 1
            continue

        # Inside a quoted region, preserve escape sequences and quote doubling.
        if not powershell:
            if char == "\\" and i + 1 < n:
                output.append(code[i : i + 2])
                i += 2
                continue
        elif char == "`" and i + 1 < n:
            output.append(code[i : i + 2])
            i += 2
            continue
        elif char == quote and i + 1 < n and code[i + 1] == quote:
            output.append(code[i : i + 2])
            i += 2
            continue
        output.append(char)
        if char == quote:
            quote = None
        i += 1

    return "".join(output), changed


def _bash_heredoc_end(code, start):
    """Return the end of a POSIX here-document starting at ``start``, if any."""
    if not code.startswith("<<", start) or code.startswith("<<<", start):
        return None
    match = _BASH_HEREDOC_RE.match(code, start)
    if not match:
        return None
    delimiter = next(group for group in match.groups() if group is not None)
    if not delimiter:
        return None
    strip_tabs = code[start + 2] == "-"
    line_end = code.find("\n", match.end())
    if line_end == -1:
        return len(code)
    index = line_end + 1
    while index < len(code):
        end = code.find("\n", index)
        body = code[index:] if end == -1 else code[index:end]
        line = body[:-1] if body.endswith("\r") else body
        if strip_tabs:
            line = line.lstrip("\t")
        if line == delimiter:
            return len(code) if end == -1 else end + 1
        index = len(code) if end == -1 else end + 1
    return len(code)


def _powershell_here_string_end(code, start):
    """Return the end of a PowerShell here-string starting at ``start``, if any."""
    marker = None
    if code.startswith('@"', start):
        marker = '"@'
    elif code.startswith("@'", start):
        marker = "'@"
    if marker is None:
        return None
    if start > 0 and code[start - 1] not in _COMMENT_BOUNDARY_CHARS:
        return None
    index = start + 2
    while True:
        newline = code.find("\n", index)
        if newline == -1:
            return None
        if code.startswith(marker, newline + 1):
            return newline + 1 + len(marker)
        index = newline + 1


def _powershell_block_comment_end(code, start):
    """Return the end of a possibly nested PowerShell block comment."""
    depth = 0
    index = start
    while index < len(code):
        if code.startswith("<#", index):
            depth += 1
            index += 2
        elif code.startswith("#>", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    return len(code)