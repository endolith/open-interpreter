import ast

from interpreter.terminal_interface.profiles.profiles import (
    RemoveInterpreter,
    apply_profile_to_object,
)


def test_remove_interpreter_strips_import_and_assignment():
    source = (
        "from interpreter import interpreter\n"
        "interpreter = OpenInterpreter()\n"
        "x = 1\n"
    )
    tree = ast.parse(source)
    transformed = RemoveInterpreter().visit(tree)
    ast.fix_missing_locations(transformed)
    new_source = ast.unparse(transformed)
    assert "from interpreter import interpreter" not in new_source
    assert "OpenInterpreter()" not in new_source
    assert "x = 1" in new_source


def test_apply_profile_to_object_nested():
    class Inner:
        def __init__(self):
            self.temperature = 0.0

    class Outer:
        def __init__(self):
            self.llm = Inner()

    obj = Outer()
    apply_profile_to_object(obj, {"llm": {"temperature": 0.7}})
    assert obj.llm.temperature == 0.7
