#!/usr/bin/env bash
# This script recursively collects help output from "kubectl operator" subcommands.
# It writes the output to kubectl_operator_help.txt.
# The recursion starts at the base command (i.e. "kubectl operator --help").

set -euo pipefail

OUTPUT_FILE="kubectl_operator_help.txt"
: > "$OUTPUT_FILE"

declare -A VISITED

# Recursively call help for each command path.
# $1: the current command path (e.g. "catalog add")
get_help() {
    local cmd_path="$1"
    # Build the full command.
    local full_cmd="kubectl operator"
    if [[ -n "$cmd_path" ]]; then
        full_cmd+=" $cmd_path"
    fi
    full_cmd+=" --help"

    # Skip if already processed.
    if [[ -n "${VISITED["$full_cmd"]:-}" ]]; then
        return
    fi
    VISITED["$full_cmd"]=1

    # Capture help output.
    local help_text
    if ! help_text=$(eval "$full_cmd" 2>&1); then
        echo "Error running: $full_cmd" >&2
        return
    fi

    {
        echo "COMMAND: $full_cmd"
        echo "----------------------------------------"
        echo "$help_text"
        echo -e "\n\n"
    } >> "$OUTPUT_FILE"

    # Look for available subcommands in the help output.
    local in_section=0
    local sub
    while IFS= read -r line; do
        if [[ "$line" =~ ^Available\ Commands: ]]; then
            in_section=1
            continue
        fi
        if [[ $in_section -eq 1 && ( "$line" =~ ^Flags: || "$line" =~ ^Global\ Flags: || "$line" =~ ^Use\ ) ]]; then
            break
        fi
        if [[ $in_section -eq 1 ]]; then
            # Extract the first word as the subcommand.
            sub=$(echo "$line" | awk '{print $1}')
            # Skip empty strings or the help command.
            if [[ -z "$sub" || "$sub" == "help" ]]; then
                continue
            fi
            # Recurse with the new subcommand appended.
            if [[ -n "$cmd_path" ]]; then
                get_help "$cmd_path $sub"
            else
                get_help "$sub"
            fi
        fi
    done <<< "$help_text"
}

# Start recursion at the base command.
get_help ""

echo "Help output collected in $OUTPUT_FILE"
