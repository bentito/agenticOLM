#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import requests  # required for the local client

########################################################################
# 1) Local client classes that return a structure matching OpenAI's style
########################################################################

class LocalChoice:
    """Mimics a single 'choice' from OpenAI."""
    def __init__(self, message_dict):
        # message_dict is expected to have "role" and "content" and optionally "function_call"
        self.message = message_dict

class LocalChatCompletionResponse:
    """Mimics the entire 'ChatCompletion' response from OpenAI."""
    def __init__(self, raw_data):
        if isinstance(raw_data, dict) and "choices" in raw_data and isinstance(raw_data["choices"], list):
            self.choices = [LocalChoice(choice["message"]) for choice in raw_data["choices"]]
        else:
            # If not already in standard form, wrap the response into a minimal structure.
            text_part = raw_data.get("text") if isinstance(raw_data, dict) else str(raw_data)
            fallback_message = {"role": "assistant", "content": text_part}
            self.choices = [LocalChoice(fallback_message)]

class LocalChatCompletions:
    """Mimics openai.ChatCompletion for local usage."""
    def __init__(self, endpoint, model):
        self.endpoint = endpoint
        self.model = model

    def create(self, **kwargs):
        kwargs.setdefault("model", self.model)
        url = f"{self.endpoint}/v1/chat/completions"
        resp = requests.post(url, json=kwargs)
        data = resp.json()
        return LocalChatCompletionResponse(data)

class LocalChat:
    """Mimics openai.Chat for local usage."""
    def __init__(self, endpoint, model):
        self.completions = LocalChatCompletions(endpoint, model)

class LocalClient:
    """Mimics openai client usage: client.chat.completions.create(...)"""
    def __init__(self, endpoint, model):
        self.chat = LocalChat(endpoint, model)

########################################################################
# 2) Use the required flag --model <model_name> to set the local model name
########################################################################

if len(sys.argv) < 3 or sys.argv[1] != "--model":
    print("Usage: {} --model <model_name>".format(sys.argv[0]))
    sys.exit(1)

MODEL_NAME = sys.argv[2]
client = LocalClient("http://0.0.0.0:8080", MODEL_NAME)

########################################################################
# 3) KUBECTL SUBCOMMAND WRAPPERS (ACT functions)
########################################################################

def operator_catalog_add(name: str, index_image: str, display_name: str = None,
                           publisher: str = None, cleanup_timeout: str = None,
                           namespace: str = None, timeout: str = None) -> str:
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

def operator_catalog_list(all_namespaces: bool = False, namespace: str = None,
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

def operator_catalog_remove(catalog_name: str, namespace: str = None,
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

def operator_completion_fish(no_descriptions: bool = False, namespace: str = None,
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

def operator_completion_powershell(no_descriptions: bool = False, namespace: str = None,
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

def operator_completion_zsh(no_descriptions: bool = False, namespace: str = None,
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

def operator_describe(operator: str, catalog: str = None, channel: str = None,
                      with_long_description: bool = False, namespace: str = None,
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

def operator_install(operator: str, approval: str = None, channel: str = None,
                     cleanup_timeout: str = None, create_operator_group: bool = False,
                     version: str = None, watch: list = None, namespace: str = None,
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

def operator_list_available(catalog: str = None, namespace: str = None,
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

def operator_olmv1_install(operator: str, namespace: str = None,
        timeout: str = None,
) -> str:
    """Install an operator using OLMv1."""
    cmd = ["kubectl", "operator", "olmv1", "install", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_uninstall(operator: str, namespace: str = None,
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
    """Uninstall an operator (and optionally its operands and groups)."""
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

def operator_upgrade(operator: str, channel: str = None, namespace: str = None,
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


def operator_list(
        namespace: str = None,
        timeout: str = None,
) -> str:
    """List installed operators on the cluster."""
    cmd = ["kubectl", "operator", "list"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)


# -------------------------------------------------------------------------
# Helper to execute shell commands that include pipelines.
# -------------------------------------------------------------------------
def _run_shell_command(cmd_str: str) -> str:
    """Helper to execute a command string using the shell (supports pipelines)."""
    print(f"[ACT] Executing command: `{cmd_str}`")
    try:
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {cmd_str}:\n{e.stderr}"


# -------------------------------------------------------------------------
# FILTERED VERSIONS: Composite commands using grep for filtering output.
# -------------------------------------------------------------------------
def operator_catalog_list_filtered(
        filter_term: str,
        all_namespaces: bool = False,
        namespace: str = None,
        timeout: str = None,
) -> str:
    """List installed operator catalogs, filtering the output with grep."""
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

def operator_list_available_filtered(filter_term: str, catalog: str = None,
        namespace: str = None,
        timeout: str = None,
) -> str:
    """List operators available to be installed, filtering the output with grep."""
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

def operator_list_operands_filtered(operator: str, filter_term: str, output: str = None,
        namespace: str = None,
        timeout: str = None,
) -> str:
    """List operands of an installed operator, filtering the output with grep."""
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)


# -------------------------------------------------------------------------
# Helper to execute commands without pipelines.
# -------------------------------------------------------------------------
def _run_subprocess(cmd):
    """Execute a command, capturing both stdout and stderr, returning whichever contains the output or error."""
    command_str = " ".join(cmd)
    print(f"[ACT] Executing command: `{command_str}`")

    # Run the command without check=True, so we can inspect result manually.
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Combine stdout and stderr in case the CLI tool prints error text to stdout.
    combined_output = (result.stdout or "") + (result.stderr or "")

    # If returncode != 0, treat it as an error and return the combined output as error text.
    if result.returncode != 0:
        error_output = f"Error executing `{command_str}` (exit code {result.returncode}):\n{combined_output}"
        print(f"[DEBUG] {error_output}")
        return error_output

    # If command succeeded (exit code 0) but there's still something in stderr, log it as debug.
    if result.stderr.strip():
        debug_info = f"[DEBUG] Non-empty stderr on success:\n{result.stderr.strip()}"
        print(debug_info)
        # Optionally, you can append this debug info to the returned output or just ignore it.
        return result.stdout + "\n" + debug_info

    # Otherwise, just return stdout.
    return result.stdout


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
                "create_operator_group": {"type": "boolean",
                                          "description": "Whether to create an operator group if needed."},
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
                "operand_strategy": {"type": "string",
                                     "description": "Strategy to handle operands (abort, ignore, or delete)."},
                "delete_operator_groups": {"type": "boolean", "description": "Delete operator groups if applicable."},
                "delete_operator": {"type": "boolean",
                                    "description": "Delete the operator object and associated resources."},
                "delete_all": {"type": "boolean",
                               "description": "Delete all objects (equivalent to --delete-operator and --delete-operator-groups)."},
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
    # --- Added filtered versions ---
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
    elif func_name == "operator_list":
        return operator_list(**args)
    # --- Dispatch for filtered functions ---
    elif func_name == "operator_catalog_list_filtered":
        return operator_catalog_list_filtered(**args)
    elif func_name == "operator_list_available_filtered":
        return operator_list_available_filtered(**args)
    elif func_name == "operator_list_operands_filtered":
        return operator_list_operands_filtered(**args)
    else:
        return f"Unknown function: {func_name}"

########################################################################
# 5) Helper to extract message attributes (for both GPT objects and dicts)
########################################################################
def get_message_attr(msg, attr, default=""):
    try:
        return getattr(msg, attr)
    except AttributeError:
        return msg.get(attr, default)

# -------------------------------------------------------------------------
# AGENT LOOP: Think, Act, Observe paradigm with improved prompting for error handling.
# -------------------------------------------------------------------------
from colorama import init, Fore, Style
init(autoreset=True)

def main():
    print(Fore.CYAN + "Enter your request. (Type 'quit' to exit)")
    conversation = [{
        "role": "system",
        "content": (
            "You are a Kubernetes Operator assistant following a Think-Act-Observe loop. "
            "When processing a user request, first determine the best function to call from the available tools. "
            "After executing a function, carefully observe its output—including any error messages. "
                "If the error output is ambiguous or incomplete, ask clarifying questions to the user or suggest additional function calls to further diagnose the problem. "
                "If the error clearly indicates a known corrective action (for example, if it states that the namespace has no existing operator group and instructs to use --create-operator-group), "
                "then trigger that corrective function call automatically using the available tools. "
                "OLM is a well-established, open-source system that manages operator installations, so use your knowledge of OLM to correctly interpret queries."
                "Be sure to refer to the details provided in the FUNCTIONS schema when determining which corrective action to take. "
                "In all cases, your final answer should summarize what occurred, and if necessary, ask for additional clarification or instructions."
            ),
        }
    ]

    while True:
        user_input = input(Fore.GREEN + "\nUser: " + Style.RESET_ALL)
        if user_input.strip().lower() in {"quit", "exit"}:
            print(Fore.CYAN + "Bye!")
            break

        conversation.append({"role": "user", "content": user_input})

        # Ask the local model for a response using the completions API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            functions=FUNCTIONS,
            function_call="auto"
        )

        msg = response.choices[0].message
        function_call = get_message_attr(msg, "function_call", None)
        content = get_message_attr(msg, "content", "")

        if function_call:
            print(Fore.YELLOW + "\n[THINK] " + Style.RESET_ALL + "Assistant determined a function call is needed.")
            print(Fore.MAGENTA + "[ACT] " + Style.RESET_ALL + f"Executing function: {function_call['name']}")
            func_call_obj = type("FuncCall", (object,), function_call)
            tool_output = dispatch_function_call(func_call_obj)
            print(Fore.BLUE + "[OBSERVE] " + Style.RESET_ALL + f"Function output:\n{tool_output}")

            conversation.append({
                "role": "function",
                "name": function_call["name"],
                "content": tool_output,
            })

            if tool_output.startswith("Error executing"):
                debug_msg = f"[DEBUG] Full error output:\n{tool_output.strip()}"
                print(Fore.RED + debug_msg + Style.RESET_ALL)
                conversation.append({"role": "assistant", "content": debug_msg})

            followup_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation
            )
            final_msg = get_message_attr(followup_response.choices[0].message, "content", "")
            print(Fore.CYAN + "\nAssistant: " + Style.RESET_ALL + f"{final_msg}")
            conversation.append({"role": "assistant", "content": final_msg})
        else:
            print(Fore.CYAN + "\nAssistant: " + Style.RESET_ALL + content)
            conversation.append({"role": "assistant", "content": content})

if __name__ == "__main__":
    main()