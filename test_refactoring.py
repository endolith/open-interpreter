"""
Test script to verify the computer -> toolbox, terminal refactoring
Run this to ensure everything is working correctly.
"""

from interpreter import interpreter

print("=" * 60)
print("Testing Refactoring: computer -> toolbox, terminal")
print("=" * 60)

# Test 1: Verify imports and structure
print("\n1. Testing structure...")
assert hasattr(interpreter, 'toolbox'), "interpreter.toolbox should exist"
assert hasattr(interpreter, 'terminal'), "interpreter.terminal should exist"
assert hasattr(interpreter.toolbox, 'display'), "toolbox.display should exist"
assert hasattr(interpreter.toolbox, 'mouse'), "toolbox.mouse should exist"
assert hasattr(interpreter.terminal, 'languages'), "terminal.languages should exist"
print("   ✓ Structure is correct")

# Test 2: Backward compatibility - toolbox.run() should work
print("\n2. Testing backward compatibility (toolbox.run)...")
try:
    result = list(interpreter.toolbox.run("python", "print('Hello from toolbox.run()')", stream=True))
    print("   ✓ toolbox.run() works (delegates to terminal.run)")
except Exception as e:
    print(f"   ✗ toolbox.run() failed: {e}")

# Test 3: Direct terminal access
print("\n3. Testing direct terminal access...")
try:
    result = list(interpreter.terminal.run("python", "print('Hello from terminal.run()')", stream=True))
    print("   ✓ terminal.run() works")
except Exception as e:
    print(f"   ✗ terminal.run() failed: {e}")

# Test 4: Language management
print("\n4. Testing language management...")
try:
    # Both should point to the same object
    assert interpreter.toolbox.languages is interpreter.terminal.languages, \
        "toolbox.languages and terminal.languages should be the same"
    print(f"   ✓ Languages accessible via both paths: {len(interpreter.terminal.languages)} languages")
except Exception as e:
    print(f"   ✗ Language management failed: {e}")

# Test 5: Toolbox convenience functions
print("\n5. Testing toolbox convenience functions...")
try:
    # Test that toolbox functions exist
    assert callable(interpreter.toolbox.display.screenshot), "display.screenshot should be callable"
    assert callable(interpreter.toolbox.mouse.position), "mouse.position should be callable"
    assert callable(interpreter.toolbox.keyboard.write), "keyboard.write should be callable"
    print("   ✓ Toolbox functions are accessible")
except Exception as e:
    print(f"   ✗ Toolbox functions test failed: {e}")

# Test 6: Terminal methods
print("\n6. Testing terminal methods...")
try:
    assert callable(interpreter.terminal.run), "terminal.run should be callable"
    assert callable(interpreter.terminal.stop), "terminal.stop should be callable"
    assert callable(interpreter.terminal.terminate), "terminal.terminate should be callable"
    print("   ✓ Terminal methods are accessible")
except Exception as e:
    print(f"   ✗ Terminal methods test failed: {e}")

# Test 7: Verify backward compatibility alias
print("\n7. Testing backward compatibility alias...")
try:
    # Should have interpreter.computer as an alias to toolbox (for backward compatibility)
    assert hasattr(interpreter, 'computer'), \
        "interpreter.computer should exist as backward compatibility alias"
    assert interpreter.computer is interpreter.toolbox, \
        "interpreter.computer should be the same object as interpreter.toolbox"
    print("   ✓ Backward compatibility alias (interpreter.computer) works")
except AssertionError as e:
    print(f"   ✗ {e}")

# Test 8: Test that toolbox.run shortcuts work
print("\n8. Testing toolbox shortcuts...")
try:
    # These should all delegate to terminal
    assert callable(interpreter.toolbox.run), "toolbox.run should exist"
    assert callable(interpreter.toolbox.exec), "toolbox.exec should exist"
    assert callable(interpreter.toolbox.stop), "toolbox.stop should exist"
    assert callable(interpreter.toolbox.terminate), "toolbox.terminate should exist"
    print("   ✓ Toolbox shortcuts exist")
except Exception as e:
    print(f"   ✗ Toolbox shortcuts test failed: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nIf all tests passed, the refactoring is working correctly.")
print("You can also test interactively:")
print("  - Run: interpreter")
print("  - Try: 'py print(123)'")
print("  - Try: 'what tools are in your toolbox?'")
print("  - Try: 'use toolbox.web.answer to answer a question'")
