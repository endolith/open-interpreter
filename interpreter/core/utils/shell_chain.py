"""Splitting shell command lines into their chained commands.

The execution allowlist wants to match chained commands on one line — e.g.
`cd /x && ls` should satisfy an `ls` rule — and the redundant-cd stripper
shares the same chain-operator semantics. This is the shared, shell-agnostic
splitter both use, so the rules never drift apart.
"""


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