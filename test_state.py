from interpreter import interpreter

# Set it to test offline or something, or just run a simple command
interpreter.llm.model = "none" # or whatever is free, wait we can just run the python terminal directly!

from interpreter.core.terminal.languages.python import Python

python = Python(interpreter)
for output in python.run("x = 5\nimport os"):
    print(output)
print("Finished first run. Now running something else:")
for output in python.run("y = x + 1"):
    print(output)
