import os
import re


class CwdTrackingMixin:
    """Tracks a persistent shell's working directory and strips redundant ``cd`` prefixes.

    Persistent shell subprocesses (bash, PowerShell, cmd) start in the CLI's
    working directory. LLMs habitually re-emit ``cd /the/path/i'm/already/in``
    at the top of every block — often chained with ``&&``/``;``/``&`` — which
    wastes a round trip and encourages standalone-script thinking. This mixin
    removes those prefixes when they wouldn't change the shell's location.

    The real location is learned two ways: from kept ``cd`` commands, and
    authoritatively from a ``##oi_pwd##<path>`` line the language echoes just
    before its end-of-execution marker (see ``_insert_cwd_marker``), which
    self-corrects tracking if it ever desyncs.

    Subclasses configure:
      ``cd_commands``          recognized ``cd`` spellings (PowerShell adds
                               ``Set-Location``/``sl``; default ``("cd",)``)
      ``cd_ignore_case``       case-insensitive command match (PowerShell)
      ``cd_option_prefixes``   leading option tokens to skip (cmd's ``/d``)
      ``cd_chain_operators``   chaining separators kept after a stripped cd
                               (cmd has no ``;``; default ``("&&", ";", "&")``)
    """

    cd_commands = ("cd",)
    cd_ignore_case = False
    cd_option_prefixes = ()
    cd_chain_operators = ("&&", ";", "&")

    def __init__(self):
        self.cwd = os.getcwd()
        # Set by _strip_redundant_cd when it removes cd prefixes; run() yields
        # it as a short console notice so the user sees what was removed.
        self._pending_notice = None

    @property
    def _cd_re(self):
        flags = re.IGNORECASE if self.cd_ignore_case else 0
        pattern = "|".join(re.escape(cmd) for cmd in self.cd_commands)
        return re.compile(rf"^({pattern})\s+(.+)$", flags)

    @property
    def _cd_chain_re(self):
        pattern = "|".join(re.escape(op) for op in self.cd_chain_operators)
        return re.compile(rf"^({pattern})(.*)$")

    # First whitespace-delimited token is the target; a chain operator may
    # directly follow it (`cd X; cmd`). `||` is split out too so the line can be
    # recognized as a chain and deliberately kept whole (a failed cd would have
    # run the fallback). Any other trailing content (`cd X > f`) leaves the
    # line unparseable and it is left untouched.
    _CD_TARGET_RE = re.compile(r"^(\S+?)\s*((?:&&|\|\||;|&).*)?$")

    def _cwd_marker_echo(self):
        raise NotImplementedError

    def _insert_cwd_marker(self, code, end_marker):
        """Insert the ``##oi_pwd##`` echo just before the end-of-execution marker."""
        if code.endswith(end_marker):
            return code[: -len(end_marker)] + self._cwd_marker_echo() + end_marker
        return code

    def _filter_pwd_marker(self, line):
        """If ``line`` is the cwd marker, update ``self.cwd`` and return True.

        Returns False for any other line so subclass postprocessors can handle it.
        """
        m = re.match(r"^##oi_pwd##(.*)$", line.rstrip("\r\n"))
        if not m:
            return False
        self.cwd = m.group(1).strip()
        return True

    def line_postprocessor(self, line):
        if self._filter_pwd_marker(line):
            return None  # discard the marker from visible output
        return self._postprocess_line(line)

    def strip_boilerplate(self, code):
        """Return (stripped_code, notice) after removing redundant cd prefixes.

        Non-mutating: it does NOT update the tracked cwd for kept ``cd``
        commands. Called by respond() as a "peek" before the code runs — if it
        advanced ``self.cwd`` here, the actual run (preprocess_code) would
        re-strip the same code and consider those ``cd``s redundant, stripping
        commands that genuinely change directories. Only the real run tracks
        the cwd.
        """
        self._pending_notice = None
        stripped = self._strip_redundant_cd(code, track=False)
        return stripped, self._pending_notice

    def _postprocess_line(self, line):
        return line

    def _strip_redundant_cd(self, code, track=True):
        """Drop ``cd`` prefixes that wouldn't change the shell's working directory.

        Handles the patterns LLMs actually emit: a standalone ``cd`` to the
        current directory, or ``cd <current-dir>`` chained with ``&&``, ``;`` or
        ``&`` (the ``cd`` prefix is removed and the rest of the line kept).
        ``cd .``-style no-ops, ``cd -`` and ``cd X || fallback`` chains are left
        alone, as are any other targets that don't resolve to the current dir.

        ``track`` gates the side effect of advancing ``self.cwd`` for kept
        ``cd`` commands: pass False when peeking (respond), True during the
        actual run (preprocess_code) so the tracked cwd follows the shell.
        """
        removed = []
        self._pending_notice = None
        kept_lines = []
        for line in code.split("\n"):
            target, after, ok = self._parse_cd(line)
            if ok and self._is_redundant_cd_target(target, track=track):
                chain = self._cd_chain_re.match(after) if after else None
                if chain:
                    remainder = chain.group(2).strip()
                    if remainder:
                        if track:
                            # The kept chain fragment may itself begin with a cd
                            # (e.g. `cd <cwd> && cd sub && ls` → `cd sub && ls`).
                            # Advance tracking past it so later lines in the same
                            # block are judged against the real shell cwd.
                            rtarget, _, rok = self._parse_cd(remainder)
                            if rok:
                                self._is_redundant_cd_target(rtarget, track=True)
                        kept_lines.append(remainder)
                        removed.append(target)
                    else:
                        # `cd X &&` with nothing after — the chain operator is
                        # a bash syntax error, so the whole line goes; count it
                        # so the removal notice stays accurate.
                        removed.append(target)
                elif not after:
                    removed.append(target)
                    continue  # standalone redundant cd — drop the whole line
                else:
                    kept_lines.append(line)  # cd followed by something unhandled
            else:
                kept_lines.append(line)
        if removed:
            distinct = list(dict.fromkeys(removed))[:4]
            self._pending_notice = (
                f"Removed redundant cd {', '.join(distinct)} (already in that directory)."
            )
        return "\n".join(kept_lines)

    def _parse_cd(self, line):
        """Split ``<cd-cmd> <target>…`` into (target, rest, ok), handling quoted targets."""
        m = self._cd_re.match(line)
        if not m:
            return None, "", False
        rest = m.group(2)
        for opt in self.cd_option_prefixes:
            if rest == opt:
                return None, "", False
            if rest.startswith(opt + " "):
                rest = rest[len(opt) :].lstrip()
        if rest[0] in "\"'":
            quote = rest[0]
            end = rest.find(quote, 1)
            if end == -1:
                return None, "", False
            target = rest[1:end]
            after = rest[end + 1 :].strip()
        else:
            m2 = self._CD_TARGET_RE.match(rest)
            if not m2:
                return None, "", False
            target = m2.group(1)
            after = (m2.group(2) or "").strip()
        return target, after, True

    def _is_redundant_cd_target(self, target, track=True):
        if target in (".", "./"):
            return False  # trivial no-op spelling — never emitted, never stripped
        if target == "-":
            return False  # `cd -` goes to OLDPWD, which we don't track
        if target in ("$PWD", "%CD%"):
            # Explicit "already here" markers: `$PWD` (bash/PS) and `%CD%` (cmd)
            # always mean the shell's current dir. They can't be resolved from
            # the client's env, so special-case them.
            return True
        expanded = os.path.expandvars(os.path.expanduser(target))
        resolved = self._resolve_target(expanded)
        if resolved == self._resolve_target(self.cwd):
            return True
        # Only trust the new cwd if the directory exists, so a failed `cd`
        # doesn't desync our tracking from the shell's real state. When
        # ``track`` is False (respond's peek), never advance the cwd — the cd
        # hasn't actually run yet.
        if track and os.path.isdir(resolved):
            self.cwd = resolved
        return False

    def _resolve_target(self, path):
        joined = os.path.abspath(os.path.join(self.cwd, path))
        return os.path.normcase(os.path.normpath(joined))
