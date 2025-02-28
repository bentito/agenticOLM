#!/usr/bin/env python3
import os
import json
import subprocess
import re
import requests
from colorama import init, Fore, Style

init(autoreset=True)

# ------------------------------------------------------------------------------
# CONFIGURATION FOR LOCAL MODEL
# ------------------------------------------------------------------------------
MODEL_NAME = "NousResearch-DeepHermes-3-Llama-3-8B"  # local model name
LOCAL_MODEL_URL = "http://0.0.0.0:8080/v1/chat/completions"
DEBUG = False  # Set to True to enable debug logging

# ------------------------------------------------------------------------------
# FUNCTION SCHEMAS (available tools)
# ------------------------------------------------------------------------------
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
        "name": "operator_list",
        "description": "List installed operators on the cluster (OLM). Use this function when checking if an operator is installed.",
        "parameters": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "Namespace scope (optional)."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."}
            }
        }
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
    # --- Filtered Versions ---
    {
        "name": "operator_catalog_list_filtered",
        "description": "List installed operator catalogs, filtering the output with grep.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_term": {"type": "string", "description": "Substring to filter output using grep."},
                "all_namespaces": {"type": "boolean", "description": "List catalogs in all namespaces."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["filter_term"],
        },
    },
    {
        "name": "operator_list_available_filtered",
        "description": "List operators available to be installed, filtering the output with grep.",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_term": {"type": "string", "description": "Substring to filter output using grep."},
                "catalog": {"type": "string", "description": "Catalog to query (optional)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["filter_term"],
        },
    },
    {
        "name": "operator_list_operands_filtered",
        "description": "List operands of an installed operator, filtering the output with grep.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {"type": "string", "description": "Operator whose operands to list."},
                "filter_term": {"type": "string", "description": "Substring to filter output using grep."},
                "output": {"type": "string", "description": "Output format (e.g., json or yaml)."},
                "namespace": {"type": "string", "description": "Namespace scope."},
                "timeout": {"type": "string", "description": "Time to wait before giving up."},
            },
            "required": ["operator", "filter_term"],
        },
    },
]

# ------------------------------------------------------------------------------
# COMMAND WRAPPERS (ACT functions)
# ------------------------------------------------------------------------------
def _run_subprocess(cmd):
    """Execute a command, capturing stdout/stderr."""
    command_str = " ".join(cmd)
    print(f"[ACT] Executing command: `{command_str}`")
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        error_output = f"Error executing `{command_str}` (exit code {result.returncode}):\n{combined_output}"
        print(f"[DEBUG] {error_output}")
        return error_output
    if result.stderr.strip():
        debug_info = f"[DEBUG] Non-empty stderr on success:\n{result.stderr.strip()}"
        print(debug_info)
        return result.stdout + "\n" + debug_info
    return result.stdout

def operator_catalog_add(name: str, index_image: str, display_name: str = None,
                         publisher: str = None, cleanup_timeout: str = None,
                         namespace: str = None, timeout: str = None) -> str:
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

def operator_catalog_list(all_namespaces: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_catalog_remove(catalog_name: str, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "catalog", "remove", catalog_name]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_bash(no_descriptions: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "bash"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_fish(no_descriptions: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "fish"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_powershell(no_descriptions: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "powershell"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_zsh(no_descriptions: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "zsh"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_describe(operator: str, catalog: str = None, channel: str = None,
                      with_long_description: bool = False, namespace: str = None, timeout: str = None) -> str:
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

def operator_install(operator: str, approval: str = None, channel: str = None,
                     cleanup_timeout: str = None, create_operator_group: bool = False,
                     version: str = None, watch: list = None, namespace: str = None, timeout: str = None) -> str:
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

def operator_list_available(catalog: str = None, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_list_operands(operator: str, output: str = None, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_install(operator: str, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "olmv1", "install", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_uninstall(operator: str, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "olmv1", "uninstall", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_uninstall(operator: str, operand_strategy: str = None, delete_operator_groups: bool = False,
                       delete_operator: bool = False, delete_all: bool = False, namespace: str = None, timeout: str = None) -> str:
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

def operator_upgrade(operator: str, channel: str = None, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "upgrade", operator]
    if channel:
        cmd.extend(["-c", channel])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_version(namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "version"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_list(namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

# Filtered command wrappers using grep.
def operator_catalog_list_filtered(filter_term: str, all_namespaces: bool = False, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_subprocess(["bash", "-c", cmd_str])

def operator_list_available_filtered(filter_term: str, catalog: str = None, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_subprocess(["bash", "-c", cmd_str])

def operator_list_operands_filtered(operator: str, filter_term: str, output: str = None, namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_subprocess(["bash", "-c", cmd_str])

# ------------------------------------------------------------------------------
# DISPATCH FUNCTION CALL (ACT step)
# ------------------------------------------------------------------------------
def dispatch_function_call(func_call):
    """
    Dispatch the function call by parsing its arguments (if needed) and calling the appropriate subcommand.
    """
    func_name = func_call["name"]
    args_obj = func_call["arguments"]
    # If arguments is a string, parse it; if it's already a dict, use it directly.
    if isinstance(args_obj, str):
        args = json.loads(args_obj)
    else:
        args = args_obj

    # Dispatch based on function name
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
    elif func_name == "operator_list":
        return operator_list(**args)
    # Filtered functions
    elif func_name == "operator_catalog_list_filtered":
        return operator_catalog_list_filtered(**args)
    elif func_name == "operator_list_available_filtered":
        return operator_list_available_filtered(**args)
    elif func_name == "operator_list_operands_filtered":
        return operator_list_operands_filtered(**args)
    else:
        return f"Unknown function: {func_name}"

# ------------------------------------------------------------------------------
# HELPER: Call the local model
# ------------------------------------------------------------------------------
def get_completion(messages, functions=None, function_call="auto"):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if functions:
        payload["functions"] = functions
    if function_call:
        payload["function_call"] = function_call

    if DEBUG:
        print(Fore.CYAN + "[DEBUG] Payload being sent to local model:" + Style.RESET_ALL)
        print(json.dumps(payload, indent=2))
    try:
        response = requests.post(LOCAL_MODEL_URL, json=payload)
        response.raise_for_status()
        raw_response = response.json()
        if DEBUG:
            print(Fore.CYAN + "[DEBUG] Raw response from local model:" + Style.RESET_ALL)
            print(json.dumps(raw_response, indent=2))
        return raw_response
    except Exception as e:
        print(Fore.RED + f"Error calling local model: {e}" + Style.RESET_ALL)
        return None

# ------------------------------------------------------------------------------
# AGENT LOOP: Think, Act, Observe
# ------------------------------------------------------------------------------

def parse_tool_call(content: str) -> dict:
    """
    Extract the JSON object inside <tool_call>...</tool_call> tags from the content.
    """
    pattern = r"<tool_call>\s*(\{.*\})\s*</tool_call>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            print(f"[DEBUG] Failed to parse tool call JSON: {e}")
    return None


def main():
    # Build the complete system message with tool calling instructions and agent behavior.
    system_message = (
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. "
        "You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. "
        "Here are the available tools: <tools> " + json.dumps(FUNCTIONS) + " </tools> "
        "Use the following pydantic model json schema for each tool call you will make: "
        "{\"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"], \"title\": \"FunctionCall\", \"type\": \"object\"} "
        "For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n"
        "<tool_call>\n"
        "{\"arguments\": <args-dict>, \"name\": <function-name>}\n"
        "</tool_call><|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "You are a Kubernetes Operator assistant following a Think-Act-Observe loop. "
        "When processing a user request, first determine the best function to call from the available tools. "
        "After executing a function, carefully observe its output — including any error messages. "
        "If the error output is ambiguous or incomplete, ask clarifying questions or suggest additional function calls. "
        "If the error indicates a known corrective action, trigger that corrective function call automatically. "
        "After executing a function, carefully observe its output—including any error messages—and then produce a natural language summary. "
        "Do not include any internal function call instructions or JSON in your final answer."
        "Use your knowledge of OLM to correctly interpret queries and provide a final answer summarizing what occurred."
    )

    conversation = [
        {"role": "system", "content": system_message}
    ]

    print(Fore.CYAN + "Enter your request. (Type 'quit' to exit)")
    while True:
        user_input = input(Fore.GREEN + "\nUser: " + Style.RESET_ALL)
        if user_input.strip().lower() in {"quit", "exit"}:
            print(Fore.CYAN + "Bye!")
            break

        conversation.append({"role": "user", "content": user_input})
        response = get_completion(conversation, functions=FUNCTIONS, function_call="auto")
        if response is None:
            print(Fore.RED + "Failed to get a response from the local model." + Style.RESET_ALL)
            continue

        msg = response["choices"][0]["message"]

        # If the local model doesn't set a separate function_call field,
        # try to parse it from the content wrapped in <tool_call> tags.
        if "function_call" not in msg:
            parsed_call = parse_tool_call(msg.get("content", ""))
            if parsed_call:
                msg["function_call"] = parsed_call

        if "function_call" in msg:
            print(Fore.YELLOW + "\n[THINK] " + Style.RESET_ALL + "Assistant determined a function call is needed.")
            print(Fore.MAGENTA + "[ACT] " + Style.RESET_ALL + f"Executing function: {msg['function_call']['name']}")
            tool_output = dispatch_function_call(msg["function_call"])
            print(Fore.BLUE + "[OBSERVE] " + Style.RESET_ALL + f"Function output:\n{tool_output}")
            conversation.append({
                "role": "function",
                "name": msg["function_call"]["name"],
                "content": tool_output,
            })

            if tool_output.startswith("Error executing"):
                debug_msg = f"[DEBUG] Full error output:\n{tool_output.strip()}"
                print(Fore.RED + debug_msg + Style.RESET_ALL)
                conversation.append({
                    "role": "assistant",
                    "content": debug_msg
                })

            followup_response = get_completion(conversation)
            if followup_response is None:
                print(Fore.RED + "Failed to get a follow-up response from the local model." + Style.RESET_ALL)
                continue
            final_msg = followup_response["choices"][0]["message"]["content"]
            print(Fore.CYAN + "\nAssistant: " + Style.RESET_ALL + f"{final_msg}")
            conversation.append({"role": "assistant", "content": final_msg})
        else:
            print(Fore.CYAN + "\nAssistant: " + Style.RESET_ALL + f"{msg.get('content', '')}")
            conversation.append({"role": "assistant", "content": msg.get("content", "")})
if __name__ == "__main__":
    main()