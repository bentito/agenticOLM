#!/usr/bin/env python3
import subprocess
import yaml
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    HfApiModel,
    load_tool,
    tool,
    FinalAnswerTool,
    GradioUI,
)


@tool
def use_olm_tool(subcommand: str) -> str:
    """
    Executes the kubectl operator command with the provided subcommand and returns its output.

    Args:
        subcommand: A string representing the kubectl operator subcommand to execute.
                    This long docstring contains the full help documentation for the tool.

    Full Help Documentation:
    COMMAND: kubectl operator --help
    ----------------------------------------
    Manage operators in a cluster from the command line.

    kubectl operator helps you manage operator installations in your
    cluster. It can install and uninstall operator catalogs, list
    operators available for installation, and install and uninstall
    operators from the installed catalogs.

    Usage:
      operator [command]

    Available Commands:
      catalog        Manage operator catalogs
      completion     Generate the autocompletion script for the specified shell
      describe       Describe an operator
      help           Help about any command
      install        Install an operator
      list           List installed operators
      list-available List operators available to be installed
      list-operands  List operands of an installed operator
      olmv1          Manage operators via OLMv1 in a cluster from the command line
      uninstall      Uninstall an operator and operands
      upgrade        Upgrade an operator
      version        Print version information

    Flags:
      -h, --help               help for operator
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    Use "operator [command] --help" for more information about a command.

    COMMAND: kubectl operator catalog --help
    ----------------------------------------
    Manage operator catalogs

    Usage:
      operator catalog [command]

    Available Commands:
      add         Add an operator catalog
      list        List installed operator catalogs
      remove      Remove an operator catalog

    Flags:
      -h, --help   help for catalog

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    Use "operator catalog [command] --help" for more information about a command.

    COMMAND: kubectl operator catalog add --help
    ----------------------------------------
    Add an operator catalog

    Usage:
      operator catalog add <name> <index_image> [flags]

    Flags:
          --cleanup-timeout duration   the amount of time to wait before cancelling cleanup (default 1m0s)
      -d, --display-name string        display name of the index
      -h, --help                       help for add
      -p, --publisher string           publisher of the index

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator catalog list --help
    ----------------------------------------
    List installed operator catalogs

    Usage:
      operator catalog list [flags]

    Flags:
      -A, --all-namespaces   list catalogs in all namespaces
      -h, --help             help for list

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator catalog remove --help
    ----------------------------------------
    Remove a operator catalog

    Usage:
      operator catalog remove <catalog_name> [flags]

    Flags:
      -h, --help   help for remove

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator completion --help
    ----------------------------------------
    Generate the autocompletion script for operator for the specified shell.
    See each sub-command's help for details on how to use the generated script.

    Usage:
      operator completion [command]

    Available Commands:
      bash        Generate the autocompletion script for bash
      fish        Generate the autocompletion script for fish
      powershell  Generate the autocompletion script for powershell
      zsh         Generate the autocompletion script for zsh

    Flags:
      -h, --help   help for completion

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    Use "operator completion [command] --help" for more information about a command.

    COMMAND: kubectl operator completion bash --help
    ----------------------------------------
    Generate the autocompletion script for the bash shell.

    This script depends on the 'bash-completion' package.
    If it is not installed already, you can install it via your OS's package manager.

    To load completions in your current shell session:

    	source <(operator completion bash)

    To load completions for every new session, execute once:

    #### Linux:

    	operator completion bash > /etc/bash_completion.d/operator

    #### macOS:

    	operator completion bash > $(brew --prefix)/etc/bash_completion.d/operator

    You will need to start a new shell for this setup to take effect.

    Usage:
      operator completion bash

    Flags:
      -h, --help              help for bash
          --no-descriptions   disable completion descriptions

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator completion fish --help
    ----------------------------------------
    Generate the autocompletion script for the fish shell.

    To load completions in your current shell session:

    	operator completion fish | source

    To load completions for every new session, execute once:

    	operator completion fish > ~/.config/fish/completions/operator.fish

    You will need to start a new shell for this setup to take effect.

    Usage:
      operator completion fish [flags]

    Flags:
      -h, --help              help for fish
          --no-descriptions   disable completion descriptions

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator completion powershell --help
    ----------------------------------------
    Generate the autocompletion script for powershell.

    To load completions in your current shell session:

    	operator completion powershell | Out-String | Invoke-Expression

    To load completions for every new session, add the output of the above command
    to your powershell profile.

    Usage:
      operator completion powershell [flags]

    Flags:
      -h, --help              help for powershell
          --no-descriptions   disable completion descriptions

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator completion zsh --help
    ----------------------------------------
    Generate the autocompletion script for the zsh shell.

    If shell completion is not already enabled in your environment you will need
    to enable it.  You can execute the following once:

    	echo "autoload -U compinit; compinit" >> ~/.zshrc

    To load completions in your current shell session:

    	source <(operator completion zsh)

    To load completions for every new session, execute once:

    #### Linux:

    	operator completion zsh > "${fpath[1]}/_operator"

    #### macOS:

    	operator completion zsh > $(brew --prefix)/share/zsh/site-functions/_operator

    You will need to start a new shell for this setup to take effect.

    Usage:
      operator completion zsh [flags]

    Flags:
      -h, --help              help for zsh
          --no-descriptions   disable completion descriptions

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator describe --help
    ----------------------------------------
    Describe an operator

    Usage:
      operator describe <operator> [flags]

    Flags:
      -c, --catalog NamespacedNameValue   catalog to query (default: search all cluster catalogs) (default /)
      -C, --channel string                package channel to describe
      -h, --help                          help for describe
      -L, --with-long-description         include long description

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator install --help
    ----------------------------------------
    Install an operator

    Usage:
      operator install <operator> [flags]

    Flags:
      -a, --approval ApprovalValue     approval (Manual or Automatic) (default Manual)
      -c, --channel string             subscription channel
          --cleanup-timeout duration   the amount of time to wait before cancelling cleanup (default 1m0s)
      -C, --create-operator-group      create operator group if necessary
      -h, --help                       help for install
      -v, --version string             install specific version for operator (default latest)
      -w, --watch strings              namespaces to watch

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator list --help
    ----------------------------------------
    List installed operators

    Usage:
      operator list [flags]

    Flags:
      -A, --all-namespaces   list operators in all namespaces
      -h, --help             help for list

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator list-available --help
    ----------------------------------------
    List operators available to be installed

    Usage:
      operator list-available [flags]

    Flags:
      -c, --catalog NamespacedNameValue   catalog to query (default: search all cluster catalogs) (default /)
      -h, --help                          help for list-available

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator list-operands --help
    ----------------------------------------
    List operands of an installed operator.

    This command lists all operands found throughout the cluster for the operator
    specified on the command line. Since the scope of an operator is restricted by
    its operator group, the output will include namespace-scoped operands from the
    operator group's target namespaces and all cluster-scoped operands.

    To search for operands for an operator in a different namespace, use the
    --namespace flag. By default, the namespace from the current context is used.

    Operand kinds are determined from the owned CustomResourceDefinitions listed in
    the operator's ClusterServiceVersion.

    Usage:
      operator list-operands <operator> [flags]

    Flags:
      -h, --help            help for list-operands
      -o, --output string   Output format. One of: json|yaml

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator olmv1 --help
    ----------------------------------------
    Manage operators via OLMv1 in a cluster from the command line.

    Usage:
      operator olmv1 [command]

    Available Commands:
      install     Install an operator
      uninstall   Uninstall an operator

    Flags:
      -h, --help   help for olmv1

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    Use "operator olmv1 [command] --help" for more information about a command.

    COMMAND: kubectl operator olmv1 install --help
    ----------------------------------------
    Install an operator

    Usage:
      operator olmv1 install <operator> [flags]

    Flags:
      -h, --help   help for install

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator olmv1 uninstall --help
    ----------------------------------------
    Uninstall deletes the named Operator object.

    Warning: this command permanently deletes objects from the cluster. If the
    uninstalled Operator bundle contains CRDs, the CRDs will be deleted, which
    cascades to the deletion of all operands.

    Usage:
      operator olmv1 uninstall <operator> [flags]

    Flags:
      -h, --help   help for uninstall

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator uninstall --help
    ----------------------------------------
    Uninstall removes the subscription, operator and optionally operands managed
    by the operator as well as the relevant operatorgroup.

    Warning: this command permanently deletes objects from the cluster. Running
    uninstall concurrently with other operations could result in undefined behavior.

    The uninstall command first checks to find the subscription associated with the
    operator. It then lists all operands found throughout the cluster for the
    operator specified if one is found. Since the scope of an operator is restricted
    by its operator group, this search will include namespace-scoped operands from the
    operator group's target namespaces and all cluster-scoped operands.

    The operand-deletion strategy is then considered if any operands are found
    on-cluster. One of abort|ignore|delete. By default, the strategy is "abort",
    which means that if any operands are found when deleting the operator abort the
    uninstall without deleting anything. The "ignore" strategy keeps the operands on
    cluster and deletes the subscription and the operator. The "delete" strategy
    deletes the subscription, operands, and after they have finished finalizing, the
    operator itself.

    Setting --delete-operator-groups to true will delete the operatorgroup in the
    provided namespace if no other active subscriptions are currently in that
    namespace, after removing the operator. The subscription and operatorgroup will
    be removed even if the operator is not found.

    There are other deletion flags for removing additional objects, such as custom
    resource definitions, operator objects, and any other objects deployed alongside
    the operator (e.g. RBAC objects for the operator). These flags are:

      --delete-operator

          Deletes all objects associated with the operator by looking up the
          operator object for the operator and deleting every referenced object
          and then deleting the operator object itself. This implies the flag
          '--operand-strategy=delete' because it is impossible to delete CRDs
          without also deleting instances of those CRDs.

      -X, --delete-all

          This is a convenience flag that is effectively equivalent to the flags
          '--delete-operator=true --delete-operator-groups=true'.

    NOTE: This command does not recursively uninstall unused dependencies. To return
    a cluster to its state prior to a 'kubectl operator install' call, each
    dependency of the operator that was installed automatically by OLM must be
    individually uninstalled.

    COMMAND: kubectl operator upgrade --help
    ----------------------------------------
    Upgrade an operator

    Usage:
      operator upgrade <operator> [flags]

    Flags:
      -c, --channel string   subscription channel
      -h, --help             help for upgrade

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)

    COMMAND: kubectl operator version --help
    ----------------------------------------
    Print version information

    Usage:
      operator version [flags]

    Flags:
      -h, --help   help for version

    Global Flags:
      -n, --namespace string   If present, namespace scope for this CLI request
          --timeout duration   The amount of time to wait before giving up on an operation. (default 1m0s)
    """
    cmd = ["kubectl", "operator"] + subcommand.split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {' '.join(cmd)}:\n{e.stderr}"


# Create the model instance.
model = HfApiModel(
    max_tokens=4096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)

# Load prompt configuration from prompt.yaml.
with open("prompts.yaml", "r") as f:
    prompt_templates = yaml.safe_load(f)

# Instantiate the CodeAgent with the required tools.
agent = CodeAgent(
    model=model,
    tools=[use_olm_tool, FinalAnswerTool()],  # FinalAnswerTool must always be included.
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates,
)

# Launch the Gradio UI.
GradioUI(agent).launch()