"""
Bare-kernel `introspect(obj, pattern='')` helper.

Why this exists: the model guesses method names on unfamiliar objects
(e.g. `sh.list_sheets()` on a gspread Spreadsheet; the real call is
`sh.worksheets()`). `help()` dumps unfilterable prose, so the fix is a
filtered signature lister the model reaches for *before* calling.

It prints (never returns a value, like `help()`), so the call stands alone
as a statement. `pattern` is a case-insensitive regular expression over
member names, or a list of them. Loaded into the kernel by absolute path at
startup (see JupyterLanguage.__init__), never via `import interpreter` —
importing the package constructs OpenInterpreter and would slow every
kernel boot. Stdlib only (`difflib`, `inspect`, `re`); never executes
descriptors, properties, or `__getattr__`.
"""

import difflib
import inspect
import re

_DEFAULT_CAP = 4000
_MAX_DEFAULT_REPR_LEN = 40
_SECRET_HINTS = (
    "token",
    "secret",
    "passwd",
    "password",
    "api_key",
    "apikey",
    "auth",
    "credential",
    "private_key",
    "access_key",
)
_DROP_FIRST_NAMES = ("self", "cls", "mcs")


def _as_patterns(pattern):
    """Normalize pattern to a list of regex strings; empty means match all."""
    if pattern is None:
        return []
    if isinstance(pattern, str):
        pattern = [pattern]
    return [str(p) for p in pattern if str(p) != ""]


def _compile(regexes):
    """Compile patterns case-insensitively; return (compiled, invalid)."""
    compiled = []
    invalid = []
    for rx in regexes:
        try:
            compiled.append(re.compile(rx, re.IGNORECASE))
        except re.error:
            invalid.append(rx)
    return compiled, invalid


def _matches(name, compiled):
    """Check whether a member name matches any compiled pattern."""
    if not compiled:
        return True
    return any(rx.search(name) for rx in compiled)


def _visible_names(obj):
    """All attribute names via dir(), including privates/dunders."""
    try:
        return list(dir(obj))
    except Exception:
        return []


def _format_default(param_name, value):
    """Render a default value, redacting secrets and truncating long reprs."""
    if any(hint in param_name.lower() for hint in _SECRET_HINTS):
        return "***"
    try:
        rendered = repr(value)
    except Exception:
        return "..."
    if len(rendered) > _MAX_DEFAULT_REPR_LEN:
        return "..."
    return rendered


def _format_annotation(annotation):
    """Render an annotation compactly, truncating anything elaborate."""
    if annotation is inspect.Parameter.empty:
        return ""
    try:
        if isinstance(annotation, type):
            text = annotation.__name__
        else:
            text = str(annotation)
    except Exception:
        return ""
    if len(text) > _MAX_DEFAULT_REPR_LEN:
        return "..."
    return text


def _format_signature(sig):
    """Render an inspect.Signature with redacted/truncated defaults."""
    parts = []
    marker_inserted = False
    posonly_count = sum(
        1
        for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_ONLY
    )
    for i, p in enumerate(sig.parameters.values()):
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append("*" + p.name)
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append("**" + p.name)
        else:
            if p.kind == inspect.Parameter.KEYWORD_ONLY and not marker_inserted:
                parts.append("*")
                marker_inserted = True
            piece = p.name
            annotation = _format_annotation(p.annotation)
            if annotation:
                piece += ": " + annotation
            if p.default is not inspect.Parameter.empty:
                piece += "=" + _format_default(p.name, p.default)
            parts.append(piece)
        if posonly_count and i == posonly_count - 1:
            parts.append("/")
    return "(" + ", ".join(parts) + ")"


def _signature_of(static, drop_first_allowed):
    """Return a display signature string, or None if not signable.

    Works on static (unexecuted) attribute values, so descriptors and
    properties are never triggered. Drops a leading self/cls only for
    functions found on a class, never for instance-held callables.
    """
    func = static
    drop_first = False
    if isinstance(static, (staticmethod, classmethod)):
        func = static.__func__
        drop_first = isinstance(static, classmethod)
    elif inspect.isfunction(func):
        drop_first = drop_first_allowed
    elif inspect.ismethoddescriptor(func) or inspect.isbuiltin(func):
        # C slot wrappers (e.g. list.append) are always methods; the
        # self/cls/mcs name check below keeps plain builtins like len safe.
        drop_first = drop_first_allowed
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return None
    params = list(sig.parameters.values())
    if (
        drop_first
        and params
        and params[0].name in _DROP_FIRST_NAMES
        and params[0].kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ):
        params = params[1:]
    try:
        return _format_signature(sig.replace(parameters=params))
    except (ValueError, TypeError):
        return None


def _is_plain_callable(obj):
    """True for a function/builtin/method, which has no member table."""
    return (
        inspect.isfunction(obj)
        or inspect.isbuiltin(obj)
        or inspect.ismethod(obj)
        or inspect.ismethoddescriptor(obj)
    )


def _static_members(obj, private=False):
    """List (name, static_value, owner, inherited) without executing anything.

    Reads class __dict__s along the MRO and the instance __dict__ directly,
    so properties, descriptors, and __getattr__ never run. Private/dunder
    names are skipped unless private=True. Returns None for a bare
    function/method, which is rendered as a single entry.
    """

    def skip(name, seen):
        return (name.startswith("_") and not private) or name in seen

    if inspect.ismodule(obj):
        return [
            (k, v, "module", False)
            for k, v in vars(obj).items()
            if not skip(k, ())
        ]
    if inspect.isclass(obj):
        out = []
        seen = set()
        for klass in obj.__mro__:
            for k, v in vars(klass).items():
                if skip(k, seen):
                    continue
                seen.add(k)
                out.append((k, v, klass.__name__, klass is not obj))
        return out
    if _is_plain_callable(obj):
        return None
    out = []
    seen = set()
    try:
        instance_dict = vars(obj)
    except TypeError:
        instance_dict = {}
    for k, v in instance_dict.items():
        if skip(k, seen):
            continue
        seen.add(k)
        out.append((k, v, "Instance", False))
    for klass in type(obj).__mro__:
        for k, v in vars(klass).items():
            if skip(k, seen):
                continue
            seen.add(k)
            out.append((k, v, klass.__name__, klass is not type(obj)))
    return out


def _header(obj):
    """Describe what was introspected, so the output is self-orienting."""
    if inspect.ismodule(obj):
        return "module %s:" % getattr(obj, "__name__", "?")
    if inspect.isclass(obj):
        return "class %s:" % getattr(obj, "__qualname__", "?")
    if _is_plain_callable(obj):
        return "function %s:" % getattr(obj, "__name__", "?")
    return "%s (instance):" % type(obj).__name__


def _doc_of(static, full):
    """Return the (full or first-line) docstring of a static member."""
    target = static
    if isinstance(static, (staticmethod, classmethod)):
        target = static.__func__
    try:
        doc = inspect.getdoc(target)
    except Exception:
        return ""
    if not doc:
        return ""
    return doc if full else doc.splitlines()[0].strip()


def _source_of(static):
    """Return source text for a static member without executing anything."""
    target = static
    if isinstance(static, (staticmethod, classmethod)):
        target = static.__func__
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return ""


def _decorate(line, static, doc, brief):
    """Append docstring text to a member line per doc/brief flags."""
    if doc:
        full = _doc_of(static, full=True)
        if full:
            return line + "\n" + full
    elif brief:
        first = _doc_of(static, full=False)
        if first:
            return line + " — " + first
    return line


def _print_single(obj, doc, brief, src):
    """Render one function/method, honouring src/doc/brief like a member."""
    name = getattr(obj, "__name__", "?")
    if src:
        source = _source_of(obj)
        print(source.rstrip() if source else "%s: <no source available>" % name)
        return
    line = "%s%s [function]" % (name, _signature_of(obj, False) or "(...)")
    print("%s\n%s" % (_header(obj), _decorate(line, obj, doc, brief)))
    print("-- 1 matched, 1 shown --")


def introspect(
    obj, pattern="", doc=False, brief=False, src=False, private=False, cap=4000
):
    """Print the object's real members/signatures matching pattern.

    Prints and returns None (like `help()`), so call it as its own
    statement. The object argument is required: pass the object you are
    about to call before writing the call — never guess names.

    pattern : case-insensitive regex on member names, or a list of them.
    doc     : True -> full docstring of each match.
    brief   : True -> first docstring line of each match.
    src     : True -> source of the single match instead of the list.
    private : True -> include _private/__dunder__ members in the listing.
    Each line is `name(signature) [Owner]` or `[Inherited: Base]`;
    non-callables render as `name: type [Owner]`. Private/dunder names are
    hidden by default (but named when a pattern hits one), so "no matches"
    is never proof a member is missing.
    """
    try:
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = _DEFAULT_CAP
        if cap <= 0:
            cap = _DEFAULT_CAP

        members = _static_members(obj, private=private)
        if members is None:
            _print_single(obj, doc, brief, src)
            return None

        regexes = _as_patterns(pattern)
        compiled, invalid = _compile(regexes)
        if invalid and not compiled:
            # Every pattern failed to compile: never fall back to match-all.
            _report_no_matches(obj, pattern, compiled, invalid, private)
            return None
        matched = [
            (k, v, o, inh)
            for (k, v, o, inh) in members
            if _matches(k, compiled)
        ]

        if src:
            if not matched:
                _report_no_matches(obj, pattern, compiled, invalid, private)
                return None
            if len(matched) > 1:
                print(
                    "introspect: src=True needs exactly one match "
                    "(got %d). Anchor the name first, e.g. "
                    "introspect(obj, '^exact_name$', src=True)." % len(matched)
                )
                return None
            k, v, o, inh = matched[0]
            source = _source_of(v)
            if not source:
                print("%s: <no source available> (C or built-in code)" % k)
            else:
                print(source.rstrip())
            return None

        if not matched:
            _report_no_matches(obj, pattern, compiled, invalid, private)
            return None

        lines = [
            _member_line(k, v, o, inh, doc, brief) for (k, v, o, inh) in matched
        ]

        header = _header(obj)
        module_doc = ""
        if inspect.ismodule(obj) and (doc or brief):
            module_doc = _doc_of(obj, full=bool(doc))
        header_block = header + (("\n" + module_doc) if module_doc else "")

        nmatched = len(lines)
        shown = nmatched
        while shown > 1 and len(header_block) + len("\n".join(lines[:shown])) + 80 > cap:
            shown -= 1
        dropped = nmatched - shown
        text = header_block + "\n" + "\n".join(lines[:shown])
        if len(text) + 80 > cap and shown == 1:
            text = text[:cap] + "\n..."
        footer = "-- %d matched, %d shown%s --" % (
            nmatched,
            shown,
            (
                ", %d more not shown — narrow the pattern" % dropped
                if dropped
                else ""
            ),
        )
        print(text + "\n" + footer)
        return None
    except Exception as e:
        print("introspect: <unreadable: %s>" % e)
        return None


def _type_name(v):
    """Best-effort type name for a non-callable member value."""
    try:
        return type(v).__name__
    except Exception:
        return "?"


def _member_line(k, v, o, inh, doc, brief):
    """Render one member line as `name(sig) [tag]` or `name: type [tag]`."""
    tag = "[Inherited: %s]" % o if inh else "[%s]" % o
    sig = _signature_of(v, o != "Instance")
    if sig is None:
        line = "%s: %s %s" % (k, _type_name(v), tag)
    else:
        line = "%s%s %s" % (k, sig, tag)
    return _decorate(line, v, doc, brief)


def _report_no_matches(obj, pattern, compiled, invalid, private=False):
    """Explain *why* there were no matches instead of a bare "no matches".

    Three cases the caller must tell apart: an invalid regex, a name that
    exists only as a hidden private/dunder (unless private=True), and a
    genuinely absent name.
    """
    try:
        target = getattr(obj, "__name__", type(obj).__name__)
    except Exception:
        target = "?"
    if invalid and not compiled:
        print(
            "introspect: no matches for %r on %s — <invalid regex: %r>. "
            "Patterns are regular expressions; escape metacharacters such "
            "as '.' as '\\\\.'." % (pattern, target, invalid[0])
        )
        return
    hidden = [
        n
        for n in _visible_names(obj)
        if n.startswith("_") and _matches(n, compiled)
    ]
    if hidden and not private:
        if len(hidden) > 12:
            names = "%s … (+%d more)" % (", ".join(hidden[:12]), len(hidden) - 12)
        else:
            names = ", ".join(hidden)
        plural = len(hidden) > 1
        print(
            "introspect: no matches for %r on %s — %s private/dunder, "
            "which introspect hides by default: %s. Reach %s with "
            "dir(obj)/getattr(obj, name), or list %s with private=True."
            % (
                pattern,
                target,
                "they exist but are" if plural else "it exists but is",
                names,
                "them" if plural else "it",
                "them" if plural else "it",
            )
        )
        return
    public_names = [n for n in _visible_names(obj) if not n.startswith("_")]
    hint = ""
    if public_names:
        needle = _as_patterns(pattern)
        probe = needle[0] if needle else ""
        try:
            close = difflib.get_close_matches(probe, public_names, n=3, cutoff=0.6)
        except Exception:
            close = []
        if close:
            hint = " Did you mean: %s?" % ", ".join(close)
    print("introspect: no matches for %r on %s.%s" % (pattern, target, hint))
