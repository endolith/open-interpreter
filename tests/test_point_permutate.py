"""Regression test for #395: permutate block size must actually vary.

OI_POINT_PERMUTATE drew a random block size and then overwrote it with
11 on the next line, so blockSize reached cv2.adaptiveThreshold as a
constant. Importing point.py pulls heavy native deps (cv2, torch,
sentence-transformers) plus an nltk download attempt, so this pins the
fix structurally: exactly one assignment to random_block_size.
"""

import ast
from pathlib import Path


def test_block_size_draw_is_not_overwritten():
    """get_element_boxes assigns random_block_size exactly once."""
    source = Path(
        "interpreter/core/computer/display/point/point.py"
    ).read_text()
    module = ast.parse(source)
    (func,) = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_element_boxes"
    ]
    assignments = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "random_block_size"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
