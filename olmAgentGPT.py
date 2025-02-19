#!/usr/bin/env python3

import os
import json
import subprocess
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>"))

# -------------------------------------------------------------------------
# 1. OPENAI CONFIG
#    Update `openai.api_key` or set ENV variable OPENAI_API_KEY before use.
# -------------------------------------------------------------------------

MODEL_NAME = "gpt-4-0613"  # The GPT-4 version that supports function calling.

# -------------------------------------------------------------------------
# 2. KUBECTL SUBCOMMAND WRAPPERS
#    Each python function calls `subprocess` to execute a specific command.
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
    """
    Add an operator catalog.
    """
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
    """
    List installed operator catalogs.
    """
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
    """
    Remove an operator catalog.
    """
    cmd = ["kubectl", "operator", "catalog", "remove", catalog_name]
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
    """
    Describe an operator.
    """
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
    """
    Install an operator.
    """
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


def operator_list(
    all_namespaces: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """
    List installed operators.
    """
    cmd = ["kubectl", "operator", "list"]
    if all_namespaces:
        cmd.append("-A")
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
    """
    List operators available to be installed.
    """
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
    output: str = None,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """
    List operands of an installed operator.
    """
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
    """
    Install an operator using OLMv1.
    """
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
    """
    Uninstall an operator using OLMv1.
    """
    cmd = ["kubectl", "operator", "olmv1", "uninstall", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])

    return _run_subprocess(cmd)


def operator_uninstall(
    operator: str,
    operand_strategy: str = None,
    delete_operator_groups: bool = False,
    delete_operator: bool = False,
    delete_all: bool = False,
    namespace: str = None,
    timeout: str = None,
) -> str:
    """
    Uninstall an operator (and optionally operands, operatorgroups, etc.).
    """
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
    """
    Upgrade an operator.
    """
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
    """
    Print operator version information.
    """
    cmd = ["kubectl", "operator", "version"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])

    return _run_subprocess(cmd)


def _run_subprocess(cmd):
    """Helper to run subprocess and return stdout or error message."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {' '.join(cmd)}:\n{e.stderr}"


# -------------------------------------------------------------------------
# 3. DEFINE FUNCTION SCHEMAS FOR OPENAI
#    Each schema matches the Python function signature above.
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
                "cleanup_timeout": {"type": "string", "description": "Cleanup timeout (e.g. 1m0s)."},
                "namespace": {"type": "string", "description": "Namespace scope for this CLI request."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["name","index_image"]
        }
    },
    {
        "name": "operator_catalog_list",
        "description": "List installed operator catalogs.",
        "parameters": {
            "type": "object",
            "properties": {
                "all_namespaces": {"type": "boolean", "description": "List catalogs in all namespaces."},
                "namespace": {"type": "string", "description": "Namespace scope for this CLI request."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            }
        }
    },
    {
        "name": "operator_catalog_remove",
        "description": "Remove an operator catalog.",
        "parameters": {
            "type": "object",
            "properties": {
                "catalog_name": {"type": "string", "description": "Name of catalog to remove."},
                "namespace": {"type": "string", "description": "Namespace scope for this CLI request."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["catalog_name"]
        }
    },
    {
        "name": "operator_describe",
        "description": "Describe an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "catalog": {"type": "string", "description": "Catalog reference if needed."},
                "channel": {"type": "string", "description": "Package channel to describe."},
                "with_long_description": {"type": "boolean", "description": "Include long description?"},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_install",
        "description": "Install an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "approval": {"type": "string", "description": "Approval (Manual or Automatic)."},
                "channel": {"type": "string", "description": "Subscription channel."},
                "cleanup_timeout": {"type": "string", "description": "Cleanup timeout."},
                "create_operator_group": {"type": "boolean", "description": "Create operator group if necessary."},
                "version": {"type": "string", "description": "Install a specific version (or latest)."},
                "watch": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Namespaces to watch (list of strings)."
                },
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_list",
        "description": "List installed operators.",
        "parameters": {
            "type": "object",
            "properties": {
                "all_namespaces": {"type": "boolean", "description": "List operators in all namespaces."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            }
        }
    },
    {
        "name": "operator_list_available",
        "description": "List operators available to be installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "catalog": {"type": "string", "description": "Catalog to query."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            }
        }
    },
    {
        "name": "operator_list_operands",
        "description": "List operands of an installed operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "output": {"type": "string", "description": "Output format (json|yaml)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_olmv1_install",
        "description": "Install an operator via OLMv1.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_olmv1_uninstall",
        "description": "Uninstall an operator via OLMv1.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_uninstall",
        "description": "Uninstall an operator (and optional operands).",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "operand_strategy": {"type": "string", "description": "One of: abort|ignore|delete."},
                "delete_operator_groups": {"type": "boolean", "description": "Delete operator group if no other subs exist?"},
                "delete_operator": {"type": "boolean", "description": "Delete operator object & associated resources."},
                "delete_all": {"type": "boolean", "description": "Equivalent to --delete-operator + --delete-operator-groups."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_upgrade",
        "description": "Upgrade an operator.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator name."},
                "channel": {"type": "string", "description": "Subscription channel to use."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator"]
        }
    },
    {
        "name": "operator_version",
        "description": "Print operator version information.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            }
        }
    },
]


# -------------------------------------------------------------------------
# 4. FUNCTION DISPATCH
#    Function that interprets the "function_call" from GPT-4, calls the
#    corresponding Python function, and returns the text result.
# -------------------------------------------------------------------------
def dispatch_function_call(func_call):
    func_name = func_call["name"]
    args = json.loads(func_call["arguments"])

    if func_name == "operator_catalog_add":
        return operator_catalog_add(**args)
    elif func_name == "operator_catalog_list":
        return operator_catalog_list(**args)
    elif func_name == "operator_catalog_remove":
        return operator_catalog_remove(**args)
    elif func_name == "operator_describe":
        return operator_describe(**args)
    elif func_name == "operator_install":
        return operator_install(**args)
    elif func_name == "operator_list":
        return operator_list(**args)
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
        return f"Unknown function name: {func_name}"


# -------------------------------------------------------------------------
# 5. EXAMPLE: SIMPLE INTERACTIVE LOOP
#    For demonstration, we prompt GPT-4 to call these subcommands.
# -------------------------------------------------------------------------
def main():
    print("Enter your request. (Type 'quit' to exit)")
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a Kubernetes Operator assistant. You know how to handle "
                "commands related to 'kubectl operator'. Use the provided function calls "
                "when the user wants to run a subcommand."
            ),
        }
    ]

    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        conversation.append({"role": "user", "content": user_input})

        # Ask GPT-4, allowing it to use function calls:
        response = client.chat.completions.create(model=MODEL_NAME,
        messages=conversation,
        functions=FUNCTIONS,
        function_call="auto")

        msg = response["choices"][0]["message"]
        if "function_call" in msg:
            # GPT-4 wants to call our function
            func_call = msg["function_call"]
            tool_response = dispatch_function_call(func_call)

            # We'll show the result to the user and append it to the conversation
            print(f"\n[Tool output]:\n{tool_response}")
            conversation.append({
                "role": "assistant",
                "content": tool_response
            })
        else:
            # Normal text response
            assistant_response = msg["content"]
            print(f"\nAssistant: {assistant_response}")
            conversation.append({
                "role": "assistant",
                "content": assistant_response
            })


if __name__ == "__main__":
    main()