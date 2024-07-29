import numpy as np
import sys
import os
import ollama
from typing import Dict, List, Optional, Tuple


# Add the `src` directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)
print(f"src_path root: {src_path}")

from utilss.rich_format_small import format_str_v2

from config import ollama_local_server, default_system_1

ollama.api_base_url = ollama_local_server

print(f"Ollama API base URL: {ollama.api_base_url}")

History = List[Tuple[str, str]]
Messages = List[Dict[str, str]]


def history_to_messages(history: History, system: str) -> Messages:
    messages = [{"role": "system", "content": system}]
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    return messages


def messages_to_history(messages: Messages) -> Tuple[str, History]:
    # print(f"Debug: Starting messages_to_history", messages)
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    # print(f"Debug: System: {system}")
    history = []
    for q, r in zip(messages[1::2], messages[2::2]):
        history.append([format_str_v2(q["content"]), r["content"]])
    # print(f"Debug: History: {history}")
    return system, history


def get_chat_response(model_name, messages):
    response = ollama.chat(model=model_name, messages=messages)
    return response


def model_chat(
    content: str, history: Optional[History], model_name: str
) -> Tuple[str, str, History]:

    if history is None:
        history = []
    system = default_system_1
    messages = history_to_messages(history, system)
    messages.append({"role": "user", "content": content})
    print("Current messages:", messages)

    # chat with ollama
    response = ollama.chat(model=model_name, messages=messages)
    # check is ollama response is correct
    print("Ollama response:", response)
    content = response["message"]["content"]
    print("ollama response content:", content)

    system, history = messages_to_history(
        messages + [{"role": "assistant", "content": content}]
    )

    print(f"system Model Chat: {system}")
    print(f"history Model Chat: {history}")

    # return content, history
    yield history, content,


# Example usage of the imported method
def example_usage():
    formatted_string = format_str_v2("some string")
    print(formatted_string)


if __name__ == "__main__":
    example_usage()
