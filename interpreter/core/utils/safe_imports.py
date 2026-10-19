"""Modules whose import is taken to be side-effect-free.

Two features need the same judgement — "is importing this module inert?" — so
the list lives in one place:

* the execution allowlist, deciding which ``import X`` lines may run unattended;
* the redundant-import stripper, deciding which repeated ``import X`` lines are
  safe to drop because the kernel already bound them.

Deliberately excludes modules whose *import* has side effects (e.g. matplotlib
picks a backend on import) and the near-universal aliased imports
(``import numpy as np``); importing a name from here is inert, but calling into
the module is still gated separately by the allowlist's call rules.
"""

SIDE_EFFECT_FREE_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "re",
        "json",
        "time",
        "datetime",
        "random",
        "math",
        "subprocess",
        "glob",
        "shutil",
        "pathlib",
        "string",
        "typing",
        "tempfile",
        "uuid",
        "hashlib",
        "base64",
        "io",
        "csv",
        "statistics",
        "secrets",
        "pprint",
        "copy",
        "warnings",
        "platform",
        "textwrap",
        "collections",
        "functools",
        "itertools",
        "logging",
        "argparse",
        "heapq",
        "bisect",
        "decimal",
        "fractions",
        "operator",
        "queue",
        "threading",
        "struct",
        "zlib",
        "gzip",
        "pickle",
        "sqlite3",
        "socket",
        "requests",
    }
)
