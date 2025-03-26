import json
import logging
import copy
import subprocess
import os
from typing import List, Dict, Any
from typing_extensions import TypedDict

# Requires langchain_openai, langchain_core, langgraph installed:
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

########################################
# Setup
########################################

logging.basicConfig(level=logging.INFO)
llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=False)


########################################
# Helpers
########################################

def clean_markdown(text: str) -> str:
    """
    Remove triple-backtick fences and extra whitespace from LLM outputs.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def merge_flags(existing_flags: List[dict], new_flags: List[dict]) -> List[dict]:
    """
    Merge flags by 'name'. If a flag with the same name exists,
    unify or combine descriptions.
    """
    merged = copy.deepcopy(existing_flags)
    for nf in new_flags:
        found_flag = None
        for ef in merged:
            if ef["name"] == nf["name"]:
                found_flag = ef
                break
        if found_flag:
            d1, d2 = found_flag.get("description", ""), nf.get("description", "")
            if d1 != d2:
                if not d1:
                    found_flag["description"] = d2
                elif not d2:
                    found_flag["description"] = d1
                else:
                    if d2 not in d1:
                        found_flag["description"] = list({d1, d2})
        else:
            merged.append(nf)
    return merged


def ensure_command_in_path(cmd: dict, path: List[str]) -> dict:
    """
    Traverse cmd["subcommands"] along 'path', creating placeholders if needed.
    Return the command dict at that path.
    """
    current = cmd
    for part in path:
        if "subcommands" not in current:
            current["subcommands"] = []
        found = None
        for sc in current["subcommands"]:
            if sc["name"] == part:
                found = sc
                break
        if not found:
            new_sub = {"name": part, "description": "", "subcommands": []}
            current["subcommands"].append(new_sub)
            found = new_sub
        current = found
    return current


########################################
# State Definition
########################################

class ToolDiscoveryState(TypedDict, total=False):
    """
    The shared state for discovering a CLI tool's commands.
    """
    tool_path: str  # e.g. "bin/kubectl-operator"
    subcommand_queue: List[List[str]]  # queue of subcommand paths to process
    processed_subcommands: List[List[str]]
    root_command: dict  # aggregator for all commands (a single root)
    done: bool
    help_output: str  # added to store help output
    return_code: int  # added to store return code


########################################
# Graph Nodes
########################################

def init_state_node(state: ToolDiscoveryState) -> dict:
    """
    Initialize the queue with the empty sub_path (root).
    Keep the existing root_command from the initial state,
    which has already been set to name=basename(tool_path).
    """
    logging.info("Initializing tool discovery state.")
    return {
        "subcommand_queue": [[]],
        "processed_subcommands": [],
        "root_command": state["root_command"],
        "done": False
    }


def call_tool_help_node(state: ToolDiscoveryState) -> dict:
    """
    Pop the first subcommand path from the queue,
    run `tool_path ...sub_path... help` via subprocess,
    store the output.
    """
    new_state = copy.deepcopy(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state

    sub_path = queue[0]
    logging.info(f"Processing subcommand path: {sub_path}")

    cmd_list = [new_state["tool_path"]] + sub_path + ["help"]
    logging.info(f"Running command: {' '.join(cmd_list)}")

    try:
        completed_proc = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=False
        )
        new_state["help_output"] = completed_proc.stdout + "\n" + completed_proc.stderr
        new_state["return_code"] = completed_proc.returncode
    except Exception as e:
        logging.error(f"Error running the tool: {e}")
        new_state["help_output"] = f"Error: {str(e)}"
        new_state["return_code"] = 1

    return new_state


def parse_tool_help_node(state: ToolDiscoveryState) -> dict:
    """
    Pass the help text to the LLM with robust instructions
    so it can extract subcommands from 'Available Commands:' or 'Usage:' sections.
    Then integrate them into the aggregator and discover new subcommands to queue.
    """
    new_state = copy.deepcopy(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state

    sub_path = queue[0]
    help_text = new_state.get("help_output", "")
    root_cmd = new_state["root_command"]

    # Enhanced prompt to coax the LLM to find subcommands, usage lines, etc.
    prompt = f"""You are an expert at extracting CLI structure from help text. 
We have a command path: {' '.join(sub_path)}. 

Your job:
1. Read the help text below carefully for "Usage:" or "Available Commands:" or "Commands:" lines 
   (and any synonyms).
2. Identify subcommands and flags. 
3. Even if the text doesn't explicitly say 'subcommands', do your best to infer them 
   from lines like 'kubectl-operator [command]' or 'install   Install something'.
4. Return a JSON with a top-level "commands" array containing exactly one object:
   {{
     "name": "<the command name for this path>",
     "description": "<short desc>",
     "subcommands": [ ... ],
     "flags": [ {{ "name": "...", "description": "..." }} ]
   }}
   - "subcommands" should be an array of objects in the same format. 
   - If this path is empty, the name is the top-level command (we've set it in code).
   - If the path is something like ["foo","bar"], the name is just "bar" (not "foo bar").
   - If you see usage lines that might indicate more subcommands, add them. 
   - If you see relevant flags, add them to "flags".

Help Output:
{help_text}
Return only a valid JSON object in the form:
{{
  "commands": [
    {{
      "name": "...",
      "description": "...",
      "subcommands": [...],
      "flags": [...]
    }}
  ]
}}
No extra commentary.
"""

    logging.info("Parsing tool help output with the LLM.")
    response = llm.invoke([HumanMessage(content=prompt)])
    content = clean_markdown(response.content)

    try:
        parsed = json.loads(content)
        commands_found = parsed.get("commands", [])
    except Exception as e:
        logging.warning(f"Failed to parse JSON from LLM: {e}")
        commands_found = []

    if not commands_found:
        # no data to merge
        logging.info("No subcommands or flags discovered.")
        return new_state

    current_cmd_data = commands_found[0]

    # If sub_path is empty => the root command
    if len(sub_path) == 0:
        # Merge the description
        desc = current_cmd_data.get("description", "")
        if desc:
            root_cmd["description"] = desc

        # Merge flags
        new_flags = current_cmd_data.get("flags", [])
        old_flags = root_cmd.get("flags", [])
        root_cmd["flags"] = merge_flags(old_flags, new_flags)

        # Merge subcommands
        if "subcommands" in current_cmd_data:
            root_cmd["subcommands"] = current_cmd_data["subcommands"]

    else:
        # sub_path is non-empty
        parent_path = sub_path[:-1]
        this_sub_name = sub_path[-1]

        # Find or create the parent
        parent_cmd = ensure_command_in_path(root_cmd, parent_path)

        # Check if parent's subcommands has an entry for this_sub_name
        sub_list = parent_cmd.get("subcommands", [])
        existing_sub = None
        for sc in sub_list:
            if sc["name"] == this_sub_name:
                existing_sub = sc
                break
        if not existing_sub:
            existing_sub = {"name": this_sub_name, "description": "", "subcommands": []}
            sub_list.append(existing_sub)

        # Merge description
        old_desc = existing_sub.get("description", "")
        new_desc = current_cmd_data.get("description", "")
        if new_desc and (new_desc != old_desc):
            if not old_desc:
                existing_sub["description"] = new_desc
            elif new_desc not in old_desc:
                existing_sub["description"] = old_desc + " / " + new_desc

        # Merge flags
        old_flags = existing_sub.get("flags", [])
        found_flags = current_cmd_data.get("flags", [])
        existing_sub["flags"] = merge_flags(old_flags, found_flags)

        # Merge subcommands
        for child_sub in current_cmd_data.get("subcommands", []):
            child_name = child_sub["name"]
            found_child = None
            for c in existing_sub["subcommands"]:
                if c["name"] == child_name:
                    found_child = c
                    break
            if not found_child:
                existing_sub["subcommands"].append(child_sub)
            else:
                found_child.update(child_sub)

    # Now discover new subcommands to queue
    def collect_subcommands(cmd_obj: dict, base_path: List[str]) -> List[List[str]]:
        new_paths = []
        for sc in cmd_obj.get("subcommands", []):
            name = sc.get("name")
            if not name:
                continue
            candidate_path = base_path + [name]
            new_paths.append(candidate_path)
            new_paths += collect_subcommands(sc, candidate_path)
        return new_paths

    newly_discovered = collect_subcommands(current_cmd_data, sub_path)

    processed = new_state["processed_subcommands"]
    for p in newly_discovered:
        if p not in queue and p not in processed and p != sub_path:
            # skip pathological self-cycle
            if len(p) >= 2 and p[-1] == p[-2]:
                logging.warning(f"Skipping potential self-cycle path: {p}")
                continue
            queue.append(p)

    # Show partial aggregator
    logging.info("Aggregated commands after parsing:")
    logging.info(json.dumps(new_state["root_command"], indent=2))

    return new_state


def mark_subcommand_processed_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    if new_state["subcommand_queue"]:
        processed_sub = new_state["subcommand_queue"].pop(0)
        new_state["processed_subcommands"].append(processed_sub)
    return new_state


def check_done_node(state: ToolDiscoveryState) -> dict:
    if not state["subcommand_queue"]:
        state["done"] = True
    else:
        state["done"] = False
    return state


# New node: trial the tool call using a function-calling LLM.
def trial_tool_call_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    tool_desc_json = json.dumps(new_state["root_command"], indent=2)
    prompt = f"""You are an expert at formulating tool calls from a tool description in JSON.
The tool description is:
{tool_desc_json}

Your task: Based on the above tool description, generate a tool call to answer the following question:
"What operators are available on the cluster?"

Return only a valid JSON object representing the tool call with keys "tool", "function", and "arguments". Do not include any extra commentary.
"""
    logging.info("Trial tool call prompt:")
    logging.info(prompt)

    trial_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, verbose=True)
    response = trial_llm.invoke([HumanMessage(content=prompt)])
    content = clean_markdown(response.content)

    logging.info("Trial tool call LLM response:")
    logging.info(content)

    try:
        new_state["tool_call"] = json.loads(content)
    except Exception as e:
        logging.warning(f"Failed to parse trial tool call JSON: {e}")
        new_state["tool_call"] = None
    return new_state


########################################
# Build the Graph
########################################

def build_graph() -> StateGraph:
    g = StateGraph(ToolDiscoveryState)

    g.add_node("init", init_state_node)
    g.add_node("call_tool_help", call_tool_help_node)
    g.add_node("parse_help", parse_tool_help_node)
    g.add_node("mark_processed", mark_subcommand_processed_node)
    g.add_node("check_done", check_done_node)
    g.add_node("trial_tool_call", trial_tool_call_node)

    g.set_entry_point("init")
    g.add_edge("init", "call_tool_help")
    g.add_edge("call_tool_help", "parse_help")
    g.add_edge("parse_help", "mark_processed")
    g.add_edge("mark_processed", "check_done")

    def condition(state: ToolDiscoveryState) -> str:
        return "finish" if state.get("done") else "continue"

    g.add_conditional_edges("check_done", condition, {
        "finish": "trial_tool_call",
        "continue": "call_tool_help"
    })
    g.add_edge("trial_tool_call", END)

    return g.compile()


########################################
# Main
########################################

if __name__ == "__main__":
    # tools we want to discover how to use, from their help systems:
    tools = ["bin/kubectl-operator"]

    # Caching: create directory if needed.
    cache_dir = "tool_info"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    graph = build_graph()
    results = []

    for tool_path in tools:
        tool_basename = os.path.basename(tool_path)  # e.g. "kubectl-operator"
        # Cache filename: replace os.sep with underscore.
        cache_filename = tool_path.replace(os.sep, "_") + "_help.json"
        cache_filepath = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_filepath):
            logging.info(f"Found cached tool info for {tool_path} at {cache_filepath}. Loading...")
            with open(cache_filepath, "r") as f:
                final_output = json.load(f)
            print("=== Final Discovered CLI Structure for", tool_path, "===")
            print(json.dumps(final_output, indent=2))
            results.append(final_output)
        else:
            # Prepare initial state for this tool
            initial_state: ToolDiscoveryState = {
                "tool_path": tool_path,
                "subcommand_queue": [],
                "processed_subcommands": [],
                # Single root command dict
                "root_command": {
                    "name": tool_basename,
                    "description": "",
                    "subcommands": []
                },
                "done": False
            }

            try:
                config = {"recursion_limit": 150, "configurable": {"thread_id": "my-thread-id"}}
                result = graph.invoke(initial_state, config)

                final_output = {
                    "tool_path": result["tool_path"],
                    "commands": [result["root_command"]],
                    "tool_call": result.get("tool_call")
                }
                print("=== Final Discovered CLI Structure for", tool_path, "===")
                print(json.dumps(final_output, indent=2))

                results.append(final_output)
                with open(cache_filepath, "w") as f:
                    json.dump(final_output, f, indent=2)
            except Exception as e:
                logging.error(f"Graph execution error for {tool_path}: {e}")