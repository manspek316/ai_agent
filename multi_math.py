import math
from pathlib import Path

import lmstudio as lms

def multiply(a: float, b: float) -> float:
    """Given two numbers a and b. Returns the product of them."""
    print(f"\n[Tool Call] multiply({a}, {b})")  # Tool usage indicator
    return a * b

def add(a: int, b: int) -> int:
    """Given two numbers a and b, returns the sum of them."""
    print(f"\n[Tool Call] add({a}, {b})")  # Tool usage indicator
    return a + b

def is_prime(n: int) -> bool:
    """Given a number n, returns True if n is a prime number."""
    print(f"\n[Tool Call] is_prime({n})")  # Tool usage indicator
    if n < 2:
        return False
    sqrt = int(math.sqrt(n))
    for i in range(2, sqrt):
        if n % i == 0:
            return False
    return True


def print_chat_history(chat):
    print("\n--- Chat History ---")
    for msg in chat._messages:
        print(f"{msg.role}: {msg.content}")
    print("--- End of Chat ---\n")



def print_fragment(fragment, round_index=0):
    # .act() supplies the round index as the second parameter
    # Setting a default value means the callback is also
    # compatible with .complete() and .respond().
    print(fragment.content, end="", flush=True)

model = lms.llm("qwen3-1.7b")

# Add the system message to instruct the model to use only one tool per round
system_message = (
    "You are a task focused AI assistant. "
    "Rules:\n"
    "- Only use one tool in each round/turn.\n"
    "- If the user request requires multiple tools, use one tool, return the result, then use the next tool in the following turn.\n"
    "- Never try to call multiple tools simultaneously.\n"
    "- Always generate a response or provide an answer after each tool call, before making another tool call.\n"
    "- Wait for the result of each tool before proceeding to the next."
)
chat = lms.Chat(system_message)

while True:
    try:
        user_input = input(">>> ")
    except EOFError:
        print()
        break
    if not user_input:
        break
    chat.add_user_message(user_input)
    print(end="", flush=True)
    model.act(
        chat,
        tools=[multiply, add, is_prime],
        on_message=chat.append,
        on_prediction_fragment=print_fragment,
        on_round_start=lambda i: print(f"\n--- Round {i + 1} started ---"),
        on_round_end=lambda i: print(f"--- Round {i + 1} ended ---"),
    )
    print_chat_history(chat)  # <-- Add this line!
    print()
