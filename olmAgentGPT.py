#!/usr/bin/env python3
import os
import json
import subprocess
from openai import OpenAI

# Instantiate the client (ensure your OPENAI_API_KEY is set)
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

def operator_catalog_remove(
    catalog_name: str,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Remove an operator catalog."""
    cmd = ["kubectl", "operator", "catalog", "remove", catalog_name]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

# Completion subcommands for generating shell autocompletion scripts.
def operator_completion_bash(
    no_descriptions: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Generate the autocompletion script for bash."""
    cmd = ["kubectl", "operator", "completion", "bash"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_fish(
    no_descriptions: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Generate the autocompletion script for fish."""
    cmd = ["kubectl", "operator", "completion", "fish"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_powershell(
    no_descriptions: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Generate the autocompletion script for powershell."""
    cmd = ["kubectl", "operator", "completion", "powershell"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_zsh(
    no_descriptions: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Generate the autocompletion script for zsh."""
    cmd = ["kubectl", "operator", "completion", "zsh"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_describe(
    operator: str,
    catalog: str = None,
    channel: str = None,
    with_long_description: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Describe an operator."""
    cmd = ["kubectl", "operator", "describe", operator]
    if catalog:
        cmd.extend(["-c", catalog])
    if channel:
        cmd.extend(["-C", channel])
    if with_long_description:
        cmd.append("-L")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_install(
    operator: str,
    approval: str = None,
    channel: str = None,
    cleanup_timeout: str = None,
    create_operator_group: bool = False,
    version: str = None,
    watch: list = None,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Install an operator."""
    cmd = ["kubectl", "operator", "install", operator]
    if approval:
        cmd.extend(["-a", approval])
    if channel:
        cmd.extend(["-c", channel])
    if cleanup_timeout:
        cmd.extend(["--cleanup-timeout", cleanup_timeout])
    if create_operator_group:
        cmd.append("-C")
    if version:
        cmd.extend(["-v", version])
    if watch:
        for ns in watch:
            cmd.extend(["-w", ns])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_list_available(
    catalog: str = None,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """List operators available to be installed."""
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_list_operands(
    operator: str,
    output: str = None,  # e.g. "json" or "yaml"
    namespace: str = None,
    timeout: str = None,
) -> str:
    """List operands of an installed operator."""
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_install(
    operator: str,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Install an operator using OLMv1."""
    cmd = ["kubectl", "operator", "olmv1", "install", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_uninstall(
    operator: str,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Uninstall an operator using OLMv1."""
    cmd = ["kubectl", "operator", "olmv1", "uninstall", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_uninstall(
    operator: str,
    operand_strategy: str = None,  # e.g. "abort", "ignore", "delete"
    delete_operator_groups: bool = False,
    delete_operator: bool = False,
    delete_all: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Uninstall an operator (and optionally operands and groups)."""
    cmd = ["kubectl", "operator", "uninstall", operator]
    if operand_strategy:
        cmd.extend(["--operand-strategy", operand_strategy])
    if delete_operator_groups:
        cmd.append("--delete-operator-groups=true")
    if delete_operator:
        cmd.append("--delete-operator=true")
    if delete_all:
        cmd.append("-X")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_upgrade(
    operator: str,
    channel: str = None,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Upgrade an operator."""
    cmd = ["kubectl", "operator", "upgrade", operator]
    if channel:
        cmd.extend(["-c", channel])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_version(
    namespace: str = None,
    timeout: str = None,
) -> str:
    """Print operator version information."""
    cmd = ["kubectl", "operator", "version"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def _run_subprocess(cmd):
    """Helper to execute a command and return its output."""
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
    {
        "name": "operator_catalog_remove",
        "description": "Remove an operator catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "catalog_name": {"type": "string", "description": "Name of the catalog to remove."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["catalog_name"],
        },
    },
    {
        "name": "operator_completion_bash",
        "description": "Generate the autocompletion script for bash.",
        "parameters": {
            "type": "object",
            "properties": {
                "no_descriptions": {"type": "boolean", "description": "Disable descriptions in completions."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    {
        "name": "operator_completion_fish",
        "description": "Generate the autocompletion script for fish.",
        "parameters": {
            "type": "object",
            "properties": {
                "no_descriptions": {"type": "boolean", "description": "Disable descriptions in completions."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    {
        "name": "operator_completion_powershell",
        "description": "Generate the autocompletion script for powershell.",
        "parameters": {
            "type": "object",
            "properties": {
                "no_descriptions": {"type": "boolean", "description": "Disable descriptions in completions."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    {
        "name": "operator_completion_zsh",
        "description": "Generate the autocompletion script for zsh.",
        "parameters": {
            "type": "object",
            "properties": {
                "no_descriptions": {"type": "boolean", "description": "Disable descriptions in completions."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    {
        "name": "operator_describe",
        "description": "Describe an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "The operator to describe."},
                "catalog": {"type": "string", "description": "Catalog to query (optional)."},
                "channel": {"type": "string", "description": "Package channel (optional)."},
                "with_long_description": {"type": "boolean", "description": "Include long description."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_install",
        "description": "Install an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name to install."},
                "approval": {"type": "string", "description": "Approval type (Manual or Automatic)."},
                "channel": {"type": "string", "description": "Subscription channel (optional)."},
                "cleanup_timeout": {"type": "string", "description": "Cleanup timeout (optional)."},
                "create_operator_group": {"type": "boolean", "description": "Whether to create an operator group if needed."},
                "version": {"type": "string", "description": "Specific version to install (or 'latest')."},
                "watch": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Namespaces to watch (optional)."
                },
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_list_available",
        "description": "List operators available to be installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "description": "Catalog to query (optional)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
    {
        "name": "operator_list_operands",
        "description": "List operands of an installed operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator whose operands to list."},
                "output": {"type": "string", "description": "Output format (e.g., json or yaml)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_olmv1_install",
        "description": "Install an operator using OLMv1.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name to install."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_olmv1_uninstall",
        "description": "Uninstall an operator using OLMv1.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name to uninstall."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_uninstall",
        "description": "Uninstall an operator (and optionally its operands and groups).",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name to uninstall."},
                "operand_strategy": {"type": "string", "description": "Strategy to handle operands (abort, ignore, or delete)."},
                "delete_operator_groups": {"type": "boolean", "description": "Delete operator groups if applicable."},
                "delete_operator": {"type": "boolean", "description": "Delete the operator object and associated resources."},
                "delete_all": {"type": "boolean", "description": "Delete all objects (equivalent to --delete-operator and --delete-operator-groups)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_upgrade",
        "description": "Upgrade an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name to upgrade."},
                "channel": {"type": "string", "description": "Subscription channel (optional)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"],
        },
    },
    {
        "name": "operator_version",
        "description": "Print operator version information.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Namespace scope (optional)."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
        },
    },
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
    elif func_name == "operator_catalog_remove":
        return operator_catalog_remove(**args)
    elif func_name == "operator_completion_bash":
        return operator_completion_bash(**args)
    elif func_name == "operator_completion_fish":
        return operator_completion_fish(**args)
    elif func_name == "operator_completion_powershell":
        return operator_completion_powershell(**args)
    elif func_name == "operator_completion_zsh":
        return operator_completion_zsh(**args)
    elif func_name == "operator_describe":
        return operator_describe(**args)
    elif func_name == "operator_install":
        return operator_install(**args)
    elif func_name == "operator_list_available":
        return operator_list_available(**args)
    elif func_name == "operator_list_operands":
        return operator_list_operands(**args)
    elif func_name == "operator_olmv1_install":
        return operator_olmv1_install(**args)
    elif func_name == "operator_olmv1_uninstall":
        return operator_olmv1_uninstall(**args)
    elif func_name == "operator_uninstall":
        return operator_uninstall(**args)
    elif func_name == "operator_upgrade":
        return operator_upgrade(**args)
    elif func_name == "operator_version":
        return operator_version(**args)
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

            # Append the function call result as an observation (Observe)
            observation = {
                "role": "function",
                "name": msg.function_call.name,
                "content": tool_output,
            }
            conversation.append(observation)
            print(f"[OBSERVE] Function output:\n{tool_output}")

            # Let the model reflect on the observation and provide a final answer
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