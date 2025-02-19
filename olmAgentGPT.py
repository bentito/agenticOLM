#!/usr/bin/env python3
import os
import json
import subprocess
from openai import OpenAI

# Instantiate a client. (The API key is read from the environment.)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>"))

MODEL_NAME = "gpt-4-0613"  # Use a model that supports function calling.

# -------------------------------------------------------------------------
# KUBECTL SUBCOMMAND WRAPPERS: Each function wraps a specific command.
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

# ... [other subcommand functions remain similar] ...

def _run_subprocess(cmd):
    """Run a subprocess command and return its output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {' '.join(cmd)}:\n{e.stderr}"

# -------------------------------------------------------------------------
# DEFINE FUNCTION SCHEMAS FOR OPENAI: Each function schema is a dict.
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
    # ... [other function schemas similar to above] ...
]

# -------------------------------------------------------------------------
# DISPATCH FUNCTION: Map function call from GPT to our local functions.
# -------------------------------------------------------------------------
def dispatch_function_call(func_call):
    func_name = func_call.name  # Use dot notation on the Pydantic model.
    args = json.loads(func_call.arguments)  # func_call.arguments is a string.
    if func_name == "operator_catalog_add":
        return operator_catalog_add(**args)
    elif func_name == "operator_catalog_list":
        return operator_catalog_list(**args)
    # ... add elif for each function ...
    else:
        return f"Unknown function: {func_name}"

# -------------------------------------------------------------------------
# INTERACTIVE LOOP: Get user input, have GPT generate a function call if needed.
# -------------------------------------------------------------------------
def main():
    print("Enter your request. (Type 'quit' to exit)")
    conversation = [
        {
            "role": "system",
            "content": ("You are a Kubernetes Operator assistant. "
                        "When appropriate, use the provided function calls "
                        "to execute kubectl operator subcommands."),
        }
    ]

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        conversation.append({"role": "user", "content": user_input})

        # Make the API call. Note that we use attribute notation to extract results.
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            functions=FUNCTIONS,
            function_call="auto"
        )

        # Instead of subscripting, use dot notation.
        msg = response.choices[0].message
        if msg.function_call:
            # GPT-4 is calling a function.
            tool_response = dispatch_function_call(msg.function_call)
            print(f"\n[Tool output]:\n{tool_response}")
            conversation.append({"role": "assistant", "content": tool_response})
        else:
            # GPT-4 is returning a normal text answer.
            assistant_response = msg.content
            print(f"\nAssistant: {assistant_response}")
            conversation.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()