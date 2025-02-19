#!/usr/bin/env python3
import os
import json
import subprocess
from openai import OpenAI

# Instantiate the client (ensure your OPENAI_API_KEY is set in your environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>"))

MODEL_NAME = "gpt-4-0613"  # Model supporting function calling

# -------------------------------------------------------------------------
# KUBECTL SUBCOMMAND WRAPPERS (ACT functions)
# -------------------------------------------------------------------------
def operator_catalog_add(
    name: str,
    index_image: str,
    display_name: str = None,
    publisher: str = None,
    cleanup_timeout: str = None,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Add an operator catalog."""
    cmd = ["kubectl", "operator", "catalog", "add", name, index_image]
    if display_name:
        cmd.extend(["-d", display_name])
    if publisher:
        cmd.extend(["-p", publisher])
    if cleanup_timeout:
        cmd.extend(["--cleanup-timeout", cleanup_timeout])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_catalog_list(
    all_namespaces: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """List installed operator catalogs."""
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

# (Other subcommand wrappers would be defined similarly)

def _run_subprocess(cmd):
    """Helper to execute a command and return output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {' '.join(cmd)}:\n{e.stderr}"

# -------------------------------------------------------------------------
# FUNCTION SCHEMAS FOR OPENAI (Descriptions for function calling)
# -------------------------------------------------------------------------
FUNCTIONS = [
    {
        "name": "operator_catalog_add",
        "description": "Add an operator catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Catalog name to add."},
                "index_image": {"type": "string", "description": "Index image URL."},
                "display_name": {"type": "string", "description": "Display name of the index."},
                "publisher": {"type": "string", "description": "Publisher of the index."},
                "cleanup_timeout": {"type": "string", "description": "Cleanup timeout (e.g., 1m0s)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["name", "index_image"],
        },
    },
    {
        "name": "operator_catalog_list",
        "description": "List installed operator catalogs.",
        "parameters": {
            "type": "object",
            "properties": {
                "all_namespaces": {"type": "boolean", "description": "List catalogs in all namespaces."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    # ... additional function schemas ...
]

# -------------------------------------------------------------------------
# DISPATCH FUNCTION CALL (ACT step)
# -------------------------------------------------------------------------
def dispatch_function_call(func_call):
    # Use dot notation (the response is now a Pydantic model)
    func_name = func_call.name
    args = json.loads(func_call.arguments)
    if func_name == "operator_catalog_add":
        return operator_catalog_add(**args)
    elif func_name == "operator_catalog_list":
        return operator_catalog_list(**args)
    # ... additional function mappings ...
    else:
        return f"Unknown function: {func_name}"

# -------------------------------------------------------------------------
# AGENT LOOP: Think, Act, Observe paradigm
# -------------------------------------------------------------------------
def main():
    print("Enter your request. (Type 'quit' to exit)")
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a Kubernetes Operator assistant. Follow a Think-Act-Observe loop: "
                "first think about the request, then act by calling the appropriate function, "
                "and finally observe its output before replying to the user."
            ),
        }
    ]

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        # Add the user message
        conversation.append({"role": "user", "content": user_input})

        # First call: Let GPT think and decide if a function call is needed
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            functions=FUNCTIONS,
            function_call="auto"  # GPT decides whether to call a function
        )
        msg = response.choices[0].message

        # Check if the model intends to call a function (Act)
        if msg.function_call:
            print("\n[THINK] Assistant determined a function call is needed.")
            print(f"[ACT] Executing function: {msg.function_call.name}")

            # Execute the function call (Act)
            tool_output = dispatch_function_call(msg.function_call)

            # Append the function call result as an observation
            observation = {
                "role": "function",
                "name": msg.function_call.name,
                "content": tool_output,
            }
            conversation.append(observation)
            print(f"[OBSERVE] Function output:\n{tool_output}")

            # Now, let the model reflect on the observation and provide a final answer
            followup_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation
            )
            final_msg = followup_response.choices[0].message.content
            print(f"\nAssistant: {final_msg}")
            conversation.append({"role": "assistant", "content": final_msg})
        else:
            # If no function call is requested, simply output the assistant's answer.
            print(f"\nAssistant: {msg.content}")
            conversation.append({"role": "assistant", "content": msg.content})

if __name__ == "__main__":
    main()