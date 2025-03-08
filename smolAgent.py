#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
import sys

from smolagents import CodeAgent, tool
from smolagents import HfApiModel, LiteLLMModel

###############################################################################
# Operator Functions (Decorated with @tool)
###############################################################################
@tool
def operator_catalog_add(name: str,
                         index_image: str,
                         display_name: str = None,
                         publisher: str = None,
                         cleanup_timeout: str = None,
                         namespace: str = None,
                         timeout: str = None) -> str:
    """
    Adds an operator catalog by executing 'kubectl operator catalog add'.

    Args:
        name: The name for the operator catalog.
        index_image: The index image URL.
        display_name: Optional display name.
        publisher: Optional publisher name.
        cleanup_timeout: Optional cleanup timeout (e.g., "1m0s").
        namespace: Optional Kubernetes namespace.
        timeout: Optional command timeout.

    Returns:
        The output of the command as a string.
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

@tool
def operator_catalog_list(all_namespaces: bool = False,
                          namespace: str = None,
                          timeout: str = None) -> str:
    """
    Lists operator catalogs using 'kubectl operator catalog list'.

    Args:
        all_namespaces: If True, list across all namespaces.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
    """
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_catalog_remove(catalog_name: str,
                            namespace: str = None,
                            timeout: str = None) -> str:
    """
    Removes an operator catalog using 'kubectl operator catalog remove'.

    Args:
        catalog_name: The name of the catalog to remove.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
    """
    cmd = ["kubectl", "operator", "catalog", "remove", catalog_name]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_completion_bash(no_descriptions: bool = False,
                             namespace: str = None,
                             timeout: str = None) -> str:
    """
    Generates a bash autocompletion script via 'kubectl operator completion bash'.

    Args:
        no_descriptions: If True, disables completion descriptions.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The autocompletion script as a string.
    """
    cmd = ["kubectl", "operator", "completion", "bash"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_completion_fish(no_descriptions: bool = False,
                             namespace: str = None,
                             timeout: str = None) -> str:
    """
    Generates a fish autocompletion script via 'kubectl operator completion fish'.

    Args:
        no_descriptions: If True, disables completion descriptions.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The autocompletion script as a string.
    """
    cmd = ["kubectl", "operator", "completion", "fish"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_completion_powershell(no_descriptions: bool = False,
                                   namespace: str = None,
                                   timeout: str = None) -> str:
    """
    Generates a PowerShell autocompletion script via 'kubectl operator completion powershell'.

    Args:
        no_descriptions: If True, disables completion descriptions.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The autocompletion script as a string.
    """
    cmd = ["kubectl", "operator", "completion", "powershell"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_completion_zsh(no_descriptions: bool = False,
                            namespace: str = None,
                            timeout: str = None) -> str:
    """
    Generates a zsh autocompletion script via 'kubectl operator completion zsh'.

    Args:
        no_descriptions: If True, disables completion descriptions.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The autocompletion script as a string.
    """
    cmd = ["kubectl", "operator", "completion", "zsh"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_describe(operator: str,
                      catalog: str = None,
                      channel: str = None,
                      with_long_description: bool = False,
                      namespace: str = None,
                      timeout: str = None) -> str:
    """
    Describes an operator via 'kubectl operator describe'.

    Args:
        operator: The operator to describe.
        catalog: Optional catalog name.
        channel: Optional subscription channel.
        with_long_description: If True, includes a long description.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The operator description as a string.
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

@tool
def operator_install(operator: str,
                     approval: str = None,
                     channel: str = None,
                     cleanup_timeout: str = None,
                     create_operator_group: bool = False,
                     version: str = None,
                     watch: list = None,
                     namespace: str = None,
                     timeout: str = None) -> str:
    """
    Installs an operator via 'kubectl operator install'.

    Args:
        operator: The operator to install.
        approval: Optional approval type ("Manual" or "Automatic").
        channel: Optional subscription channel.
        cleanup_timeout: Optional cleanup timeout.
        create_operator_group: If True, creates an operator group if needed.
        version: Optional operator version.
        watch: Optional list of namespaces to watch.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The installation command output as a string.
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

@tool
def operator_list_available(catalog: str = None,
                            namespace: str = None,
                            timeout: str = None) -> str:
    """
    Lists available operators via 'kubectl operator list-available'.

    Args:
        catalog: Optional catalog to query.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The list of available operators as a string.
    """
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_list_operands(operator: str,
                           output: str = None,
                           namespace: str = None,
                           timeout: str = None) -> str:
    """
    Lists operands for an operator via 'kubectl operator list-operands'.

    Args:
        operator: The operator whose operands to list.
        output: Optional output format (e.g., "json" or "yaml").
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The list of operands as a string.
    """
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_olmv1_install(operator: str,
                           namespace: str = None,
                           timeout: str = None) -> str:
    """
    Installs an operator using OLMv1 via 'kubectl operator olmv1 install'.

    Args:
        operator: The operator to install.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
    """
    cmd = ["kubectl", "operator", "olmv1", "install", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_olmv1_uninstall(operator: str,
                             namespace: str = None,
                             timeout: str = None) -> str:
    """
    Uninstalls an operator using OLMv1 via 'kubectl operator olmv1 uninstall'.

    Args:
        operator: The operator to uninstall.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
    """
    cmd = ["kubectl", "operator", "olmv1", "uninstall", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_uninstall(operator: str,
                       operand_strategy: str = None,
                       delete_operator_groups: bool = False,
                       delete_operator: bool = False,
                       delete_all: bool = False,
                       namespace: str = None,
                       timeout: str = None) -> str:
    """
    Uninstalls an operator via 'kubectl operator uninstall'.

    Args:
        operator: The operator to uninstall.
        operand_strategy: Optional strategy ("abort", "ignore", or "delete").
        delete_operator_groups: If True, delete associated operator groups.
        delete_operator: If True, delete the operator object.
        delete_all: If True, delete all related resources.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
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

@tool
def operator_upgrade(operator: str,
                     channel: str = None,
                     namespace: str = None,
                     timeout: str = None) -> str:
    """
    Upgrades an operator via 'kubectl operator upgrade'.

    Args:
        operator: The operator to upgrade.
        channel: Optional upgrade channel.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The command output as a string.
    """
    cmd = ["kubectl", "operator", "upgrade", operator]
    if channel:
        cmd.extend(["-c", channel])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_version(namespace: str = None,
                     timeout: str = None) -> str:
    """
    Displays operator version information via 'kubectl operator version'.

    Args:
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The version information as a string.
    """
    cmd = ["kubectl", "operator", "version"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_list(namespace: str = None,
                  timeout: str = None) -> str:
    """
    Lists installed operators via 'kubectl operator list'.

    Args:
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The list of operators as a string.
    """
    cmd = ["kubectl", "operator", "list"]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

@tool
def operator_catalog_list_filtered(filter_term: str,
                                   all_namespaces: bool = False,
                                   namespace: str = None,
                                   timeout: str = None) -> str:
    """
    Lists catalogs filtered by a term via 'kubectl operator catalog list | grep'.

    Args:
        filter_term: The term to filter catalogs.
        all_namespaces: If True, search all namespaces.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The filtered list as a string.
    """
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

@tool
def operator_list_available_filtered(filter_term: str,
                                     catalog: str = None,
                                     namespace: str = None,
                                     timeout: str = None) -> str:
    """
    Lists available operators filtered by a term via 'kubectl operator list-available | grep'.

    Args:
        filter_term: The term to filter operators.
        catalog: Optional catalog to query.
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The filtered list as a string.
    """
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

@tool
def operator_list_operands_filtered(operator: str,
                                    filter_term: str,
                                    output: str = None,
                                    namespace: str = None,
                                    timeout: str = None) -> str:
    """
    Lists operands for an operator filtered by a term via 'kubectl operator list-operands | grep'.

    Args:
        operator: The operator name.
        filter_term: The term to filter operands.
        output: Optional output format (e.g., "json" or "yaml").
        namespace: Optional namespace.
        timeout: Optional command timeout.

    Returns:
        The filtered list as a string.
    """
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

###############################################################################
# Helpers for Command Execution
###############################################################################
def _run_shell_command(cmd_str: str) -> str:
    """
    Executes a shell command string (supports pipes) and returns its output.

    Args:
        cmd_str: The full command string.

    Returns:
        The command output as a string.
    """
    print(f"[ACT] Executing command: `{cmd_str}`")
    try:
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {cmd_str}:\n{e.stderr}"

def _run_subprocess(cmd) -> str:
    """
    Executes a command (without pipes) and returns its combined stdout and stderr.

    Args:
        cmd: A list of command parts.

    Returns:
        The combined output as a string.
    """
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

###############################################################################
# CodeAgent Setup and Main Loop
###############################################################################
def create_agent(model):
    """
    Creates a CodeAgent with the provided model and operator tools.

    Args:
        model: An instance of a callable model (e.g., HfApiModel or LiteLLMModel).

    Returns:
        A CodeAgent instance.
    """
    tools = [
        operator_catalog_add,
        operator_catalog_list,
        operator_catalog_remove,
        operator_completion_bash,
        operator_completion_fish,
        operator_completion_powershell,
        operator_completion_zsh,
        operator_describe,
        operator_install,
        operator_list_available,
        operator_list_operands,
        operator_olmv1_install,
        operator_olmv1_uninstall,
        operator_uninstall,
        operator_upgrade,
        operator_version,
        operator_list,
        operator_catalog_list_filtered,
        operator_list_available_filtered,
        operator_list_operands_filtered,
    ]
    return CodeAgent(tools=tools, model=model)

def main():
    parser = argparse.ArgumentParser(
        description="Kubernetes Operator Assistant Agent (using smolagents CodeAgent)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpt", action="store_true", help="Use OpenAI (LiteLLMModel) for completions")
    group.add_argument("--local-llm", action="store_true", help="Use Hugging Face (HfApiModel) for completions")
    args = parser.parse_args()

    if args.gpt:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY environment variable")
        model = LiteLLMModel(model_id="gpt-4o-mini-2024-07-18")
        print("Using OpenAI completions (LiteLLMModel).")
    elif args.local_llm:
        # Replace with your desired Hugging Face model ID, e.g., "google/bard" or another.
        model = HfApiModel(model_id="google/bard")
        print("Using Hugging Face completions (HfApiModel).")

    agent = create_agent(model)
    print("Starting smolagents conversation. Type 'quit' to exit.")

    while True:
        try:
            user_input = input("\nUser: ")
        except (KeyboardInterrupt, EOFError):
            print("Bye!")
            break

        if user_input.strip().lower() in {"quit", "exit"}:
            print("Bye!")
            break

        result = agent.run(user_input)
        print("\nAssistant: " + result)

if __name__ == "__main__":
    main()