````sh
> interpreter

●

Welcome to Open Interpreter.

───────────────────────────────────────────────────────────────────────────────────────────

▌ OpenAI API key not found

To use gpt-4o (recommended) please provide an OpenAI API key.

To use another language model, run interpreter --local or consult the documentation at the
language models guide.

───────────────────────────────────────────────────────────────────────────────────────────

OpenAI API key: ***************************************************************************
*****************************************************************************************


Tip: To save this key for later, run one of the following and then restart your terminal.
MacOS: echo 'export OPENAI_API_KEY=your_api_key' >> ~/.zshrc
Linux: echo 'export OPENAI_API_KEY=your_api_key' >> ~/.bashrc
Windows: setx OPENAI_API_KEY your_api_key

───────────────────────────────────────────────────────────────────────────────────────────

▌ Model set to gpt-4o

Open Interpreter will require approval before running code.

Use interpreter -y to bypass this.

Press CTRL-C to exit.

> hi

  Hello! How can I assist you today?

> py hello world


  print('Hello, World!')


  Would you like to run this code? (y/n)

  y


  print('Hello, World!')


  Hello, World!


  Here is your "Hello, World!" in Python. If you have any other requests or need further
  assistance, feel free to ask!

````

