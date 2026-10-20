"""
Bare-kernel `introspect(obj, pattern='')` helper.

Why this exists: the model guesses method names on unfamiliar objects
(e.g. `sh.list_sheets()` on a gspread Spreadsheet; the real call is
`sh.worksheets()`). `help()` dumps unfilterable prose, so the fix is a
filtered signature lister the model reaches for *before* calling.

Loaded into the kernel by absolute path at startup (see
JupyterLanguage.__init__), never via `import interpreter` — importing the
package constructs OpenInterpreter and would slow every kernel boot.
Stdlib only (`difflib`, `inspect`, `re`); never executes descriptors.
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


def _matches(name, regexes):
    """Check whether a member name matches any of the pattern regexes."""
    if not regexes:
        return True
    for rx in regexes:
        try:
            if re.search(rx, name, re.IGNORECASE):
                return True
        except re.error:
            if rx.lower() in name.lower():
                return True
    return False


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


def _static_members(obj):
    """List (name, static_value, owner, inherited) without executing anything.

    Reads class __dict__s along the MRO and the instance __dict__ directly,
    so properties, descriptors, and __getattr__ never run. Returns None for
    a bare function/method, which is rendered as a single entry.
    """
    if inspect.ismodule(obj):
        return [
            (k, v, "module", False)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        ]
    if inspect.isclass(obj):
        out = []
        seen = set()
        for klass in obj.__mro__:
            for k, v in vars(klass).items():
                if k.startswith("_") or k in seen:
                    continue
                seen.add(k)
                out.append((k, v, klass.__name__, klass is not obj))
        return out
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
        return None
    out = []
    seen = set()
    try:
        instance_dict = vars(obj)
    except TypeError:
        instance_dict = {}
    for k, v in instance_dict.items():
        if k.startswith("_") or k in seen:
            continue
        seen.add(k)
        out.append((k, v, "Instance", False))
    for klass in type(obj).__mro__:
        for k, v in vars(klass).items():
            if k.startswith("_") or k in seen:
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
    if inspect.isfunction(obj) or inspect.isbuiltin(obj) or inspect.ismethod(obj):
        return "function %s:" % getattr(obj, "__name__", "?")
    return "%s (instance):" % type(obj).__name__


def _first_doc_line(static):
    """Return the first docstring line of a static member, or ''."""
    target = static
    if isinstance(static, (staticmethod, classmethod)):
        target = static.__func__
    try:
        doc = inspect.getdoc(target)
    except Exception:
        return ""
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def _full_doc(static):
    """Return the full docstring of a static member, or ''."""
    target = static
    if isinstance(static, (staticmethod, classmethod)):
        target = static.__func__
    try:
        doc = inspect.getdoc(target)
    except Exception:
        return ""
    return doc or ""


def _source_of(static):
    """Return source text for a static member without executing anything."""
    target = static
    if isinstance(static, (staticmethod, classmethod)):
        target = static.__func__
    try:
        return inspect.getsource(target)
    except (OSError, TypeError):
        return ""


def introspect(obj, pattern="", doc=False, src=False, cap=4000):
    """Print the object's real methods/signatures matching pattern.

    Prints a filtered member list and returns None (like help()). The
    object argument is required: always pass the object you are about to
    call, plus a keyword, before writing the call — never guess names.

    Each line is `name(signature) [Owner]` or `[Inherited: Base]`;
    non-callables render as `name: type [Owner]`.
    """
    try:
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = _DEFAULT_CAP
        if cap <= 0:
            cap = _DEFAULT_CAP

        members = _static_members(obj)
        if members is None:
            name = getattr(obj, "__name__", "?")
            sig = _signature_of(obj, False) or "(...)"
            print("%s\n%s%s [function]" % (_header(obj), name, sig))
            print("-- 1 matched, 1 shown --")
            return None

        regexes = _as_patterns(pattern)
        matched = [
            (k, v, o, inh)
            for (k, v, o, inh) in members
            if _matches(k, regexes)
        ]

        if src:
            if len(matched) != 1:
                print(
                    "introspect: src=True needs exactly one match "
                    "(got %d). Narrow pattern first, e.g. "
                    "introspect(obj, 'exact_name', src=True)." % len(matched)
                )
                return None
            k, v, o, inh = matched[0]
            source = _source_of(v)
            if not source:
                print("%s: <no source available>" % k)
            else:
                print(source.rstrip())
            return None

        if not matched:
            all_names = [k for (k, _v, _o, _i) in members]
            hint = ""
            if regexes:
                try:
                    close = difflib.get_close_matches(
                        regexes[0], all_names, n=3, cutoff=0.6
                    )
                except Exception:
                    close = []
                if close:
                    hint = " Did you mean: %s?" % ", ".join(close)
            print(
                "introspect: no matches for %r on %s.%s"
                % (pattern, getattr(obj, "__name__", type(obj).__name__), hint)
            )
            return None

        lines = []
        for k, v, o, inh in matched:
            tag = "[Inherited: %s]" % o if inh else "[%s]" % o
            sig = _signature_of(v, o != "Instance")
            if sig is None:
                try:
                    typename = type(v).__name__
                except Exception:
                    typename = "?"
                line = "%s: %s %s" % (k, typename, tag)
            else:
                line = "%s%s %s" % (k, sig, tag)
            if doc == "full":
                full = _full_doc(v)
                if full:
                    line += "\n" + full
            elif doc:
                first = _first_doc_line(v)
                if first:
                    line += " — " + first
            lines.append(line)

        header = _header(obj)
        nmatched = len(lines)
        text = header + "\n" + "\n".join(lines)
        shown = nmatched
        truncated = False
        while shown > 1 and len(text) + 60 > cap:
            shown -= 1
            truncated = True
            text = header + "\n" + "\n".join(lines[:shown])
        if len(text) + 60 > cap and shown <= 1:
            text = text[:cap] + "\n..."
            truncated = True
        footer = "-- %d matched, %d shown%s --" % (
            nmatched,
            shown,
            ", truncated" if truncated else "",
        )
        print(text + "\n" + footer)
        return None
    except Exception as e:
        print("introspect: <unreadable: %s>" % e)
        return None
