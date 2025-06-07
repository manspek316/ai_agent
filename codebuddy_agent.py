from pathlib import Path
import json # To parse tool arguments if not automatically handled by lmstudio

import lmstudio as lms

# --- Tool Definitions ---

def read_file(name: str):
    """
    Reads the content of a file.
    Use this tool to inspect existing code or project files.
    """
    source_path = Path(name)
    if not source_path.exists():
        return f"Error: File '{name}' not found."
    if not source_path.is_file():
        return f"Error: '{name}' is a directory, not a file."
    try:
        content = source_path.read_text(encoding="utf-8")
        return f"File '{name}' content:\n```\n{content}\n```"
    except Exception as exc:
        return f"Error reading file '{name}': {exc!r}"

def write_file(name: str, content: str):
    """
    Writes the given content to a file. If the file exists, it will be overwritten.
    Use this tool to create new code files or modify existing ones.
    """
    dest_path = Path(name)
    try:
        dest_path.write_text(content, encoding="utf-8")
        return f"File '{name}' written successfully."
    except Exception as exc:
        return f"Error writing to file '{name}': {exc!r}"

# --- LM Studio Integration and Agent Logic ---

def print_fragment(fragment, round_index=0):
    """Callback to print LLM output as it streams."""
    print(fragment.content, end="", flush=True)

# Define the system prompt for the coding agent
SYSTEM_PROMPT = """You are 'CodeBuddy', a simple, task-focused AI coding assistant.
Your primary capabilities are reading and writing files on the user's system.
You are running in a command-line environment.

**Instructions:**
1.  **Understand the Request:** Carefully read the user's request, especially noting file names and desired content.
2.  **Use Tools Judiciously:**
    * To see the content of a file, use the `read_file` tool.
    * To create a new file or modify an existing one, use the `write_file` tool.
3.  **Provide Clear Output:** After using a tool, report the outcome to the user.
4.  **Stay Focused:** Only perform actions related to reading or writing files. Do not attempt to execute code, browse the web, or perform any other actions.
5.  **Be Concise:** Your responses should be direct and to the point.
6.  **Always ask for next steps:** After completing a task or providing information, politely ask the user what they want to do next.

**Examples of user requests and your expected tool usage:**
* User: "Show me the content of `main.py`"
    Assistant (internal thought): Calls `read_file(name='main.py')`
* User: "Create a file named `hello.py` with the content `print('Hello, world!')`"
    Assistant (internal thought): Calls `write_file(name='hello.py', content='print(\\'Hello, world!\\')')`
* User: "Update `config.json` to set `debug` to `true`"
    Assistant (internal thought): First calls `read_file('config.json')`, then based on its content, calls `write_file('config.json', 'new JSON content')`
"""

def run_coding_agent():
    print("Initializing CodeBuddy AI Agent...")
    try:
        # Initialize the LM Studio model
        model = lms.llm('qwen/qwen3-8b')
    except Exception as e:
        print(f"Error initializing LM Studio model. Make sure LM Studio server is running and the model is loaded. Details: {e}")
        print("Exiting.")
        return

    # Initialize the chat with our specific system prompt
    chat = lms.Chat(SYSTEM_PROMPT)

    # Register the tools with LM Studio
    # LM Studio's .act() method expects a list of tool functions
    available_tools = [read_file, write_file]

    print("\nCodeBuddy is ready! Type your request (e.g., 'read file main.py' or 'write file hello.py with content print(\\'hello\\')').")
    print("Press Ctrl+D or leave blank to exit.")

    while True:
        try:
            user_input = input(">>> ")
        except EOFError:
            print("\nExiting CodeBuddy. Goodbye!")
            break
        if not user_input.strip():
            print("Exiting CodeBuddy. Goodbye!")
            break

        chat.add_user_message(user_input)
        print(end="", flush=True)

        try:
            # The core of the agent: let the model act with tools
            # on_message=chat.append adds the bot's full response (including tool calls) to the chat history
            # on_prediction_fragment=print_fragment streams the bot's content to the console
            model.act(
                chat,
                available_tools,
                on_message=chat.append,
                on_prediction_fragment=print_fragment,
            )
            print() # Newline after bot's streamed output
        except Exception as e:
            print(f"\nAn error occurred during model interaction: {e}")
            # Potentially add logic to clean up chat history or provide error message to LLM
            chat.add_message("system", f"An error occurred while processing your last request: {e}. Please try again or provide a different instruction.")
        print("-" * 50)

if __name__ == "__main__":
    run_coding_agent()
