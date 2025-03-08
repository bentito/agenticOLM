#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
import sys

# Import OpenAI client for GPT-based completions
from openai import OpenAI

# Import smolagents – the small agents framework
import smolagents

MODEL_NAME = "gpt-4o-mini-2024-07-18"  # Model supporting function calling and high throughput

# -------------------------------
# LLM Wrapper Classes
# -------------------------------
class GPTLLM:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_API_KEY>"))
    def chat(self, messages, functions=None, function_call="auto"):
        return self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            functions=functions,
            function_call=function_call
        )

class LocalLLM:
    def __init__(self):
        # Initialize your local LLM here (example using Hugging Face Transformers)
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        model_name = "gpt2"  # Replace with your desired local model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
    def chat(self, messages, functions=None, function_call="auto"):
        # For local LLM, use the last message as the prompt.
        prompt = messages[-1]["content"]
        output = self.pipeline(prompt, max_length=150, do_sample=True)[0]["generated_text"]
        # Mimic a response structure similar to OpenAI's response.
        class DummyResponse:
            def __init__(self, content):
                self.choices = [type("Choice", (), {"message": {"content": content, "function_call": None}})]
        return DummyResponse(output)

# -------------------------------
# Operator Functions
# -------------------------------
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

def operator_catalog_list(all_namespaces: bool = False, namespace: str = None,
                          timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "catalog", "list"]
    if all_namespaces:
        cmd.append("-A")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_catalog_remove(catalog_name: str, namespace: str = None,
                            timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "catalog", "remove", catalog_name]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_bash(no_descriptions: bool = False, namespace: str = None,
                             timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "bash"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_fish(no_descriptions: bool = False, namespace: str = None,
                             timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "fish"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_powershell(no_descriptions: bool = False, namespace: str = None,
                                    timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "completion", "powershell"]
    if no_descriptions:
        cmd.append("--no-descriptions")
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_completion_zsh(no_descriptions: bool = False, namespace: str = None,
                             timeout: str = None) -> str:
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
                      timeout: str = None) -> str:
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
                     timeout: str = None) -> str:
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
                            timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-available"]
    if catalog:
        cmd.extend(["-c", catalog])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_list_operands(operator: str, output: str = None, namespace: str = None,
                           timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_install(operator: str, namespace: str = None,
                           timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "olmv1", "install", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_olmv1_uninstall(operator: str, namespace: str = None,
                             timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "olmv1", "uninstall", operator]
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    return _run_subprocess(cmd)

def operator_uninstall(operator: str, operand_strategy: str = None,
                       delete_operator_groups: bool = False,
                       delete_operator: bool = False, delete_all: bool = False,
                       namespace: str = None, timeout: str = None) -> str:
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
                     timeout: str = None) -> str:
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

def operator_catalog_list_filtered(filter_term: str, all_namespaces: bool = False,
                                   namespace: str = None, timeout: str = None) -> str:
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
                                     namespace: str = None, timeout: str = None) -> str:
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
                                    namespace: str = None, timeout: str = None) -> str:
    cmd = ["kubectl", "operator", "list-operands", operator]
    if output:
        cmd.extend(["-o", output])
    if namespace:
        cmd.extend(["-n", namespace])
    if timeout:
        cmd.extend(["--timeout", timeout])
    cmd_str = " ".join(cmd) + f" | grep {filter_term}"
    return _run_shell_command(cmd_str)

# -------------------------------
# Helper Functions for Command Execution
# -------------------------------
def _run_shell_command(cmd_str: str) -> str:
    print(f"[ACT] Executing command: `{cmd_str}`")
    try:
        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {cmd_str}:\n{e.stderr}"

def _run_subprocess(cmd) -> str:
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

# -------------------------------
# smolagents Agent Setup
# -------------------------------
def create_agent(llm):
    # Create a dictionary mapping tool names to their corresponding functions.
    tools = {
        "operator_catalog_add": operator_catalog_add,
        "operator_catalog_list": operator_catalog_list,
        "operator_catalog_remove": operator_catalog_remove,
        "operator_completion_bash": operator_completion_bash,
        "operator_completion_fish": operator_completion_fish,
        "operator_completion_powershell": operator_completion_powershell,
        "operator_completion_zsh": operator_completion_zsh,
        "operator_describe": operator_describe,
        "operator_install": operator_install,
        "operator_list_available": operator_list_available,
        "operator_list_operands": operator_list_operands,
        "operator_olmv1_install": operator_olmv1_install,
        "operator_olmv1_uninstall": operator_olmv1_uninstall,
        "operator_uninstall": operator_uninstall,
        "operator_upgrade": operator_upgrade,
        "operator_version": operator_version,
        "operator_list": operator_list,
        "operator_catalog_list_filtered": operator_catalog_list_filtered,
        "operator_list_available_filtered": operator_list_available_filtered,
        "operator_list_operands_filtered": operator_list_operands_filtered,
    }
    agent = smolagents.Agent(
        llm=llm,
        tools=tools,
        system_prompt=(
            "You are a Kubernetes Operator assistant. Use the available tools to answer user queries. "
            "Decide on the best function to call and provide concise summaries of your actions and their outcomes."
        )
    )
    return agent

# -------------------------------
# MAIN AGENT LOOP
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Kubernetes Operator Assistant Agent (using smolagents)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpt", action="store_true", help="Use OpenAI GPT-based completions API")
    group.add_argument("--local-llm", action="store_true", help="Use local LLM for completions")
    args = parser.parse_args()

    if args.gpt:
        llm = GPTLLM()
        print("Using OpenAI GPT completions API.")
    elif args.local_llm:
        llm = LocalLLM()
        print("Using local LLM for completions.")

    agent = create_agent(llm)
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

        response = agent.act(user_input)
        print("\nAssistant: " + response)

if __name__ == "__main__":
    main()