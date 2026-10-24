"""Fail a CI test job when pytest silently ran too little or skipped a required runtime.

Usage:
    check_pytest_counts.py <junit.xml> <min-passed> [disallowed-skip-substring ...]

`min-passed` catches an all-skip (or mostly-skip) selection. Each extra
argument catches a skip reason that must not happen on that runner, e.g. a
language runtime the workflow was supposed to install.
"""

import sys
import xml.etree.ElementTree as ET


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    path = sys.argv[1]
    min_passed = int(sys.argv[2])
    disallowed = sys.argv[3:]

    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        print(f"::error::could not read {path}: {error}")
        return 1

    cases = root.findall(".//testcase")
    failures = [
        case
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    xfailed = [
        case
        for case in cases
        if case.find("skipped") is not None
        and case.find("skipped").get("type") == "pytest.xfail"
    ]
    skipped = [
        case
        for case in cases
        if case.find("skipped") is not None
        and case.find("skipped").get("type") != "pytest.xfail"
    ]
    passed = len(cases) - len(failures) - len(skipped) - len(xfailed)

    problems = []
    if passed < min_passed:
        problems.append(
            f"only {passed} tests passed (expected at least {min_passed}); "
            f"{len(skipped)} skipped, {len(failures)} failed"
        )

    for case in skipped:
        reason = (case.find("skipped").get("message") or "").replace("\n", " ")
        for needle in disallowed:
            if needle in reason:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                problems.append(f"{name} skipped: {reason}")

    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        return 1

    print(
        f"pytest guard: {passed} passed, {len(skipped)} skipped, "
        f"{len(xfailed)} xfailed, {len(failures)} failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
