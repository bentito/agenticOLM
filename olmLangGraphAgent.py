import json
import logging
import copy
import subprocess
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from typing import List, Dict, Any

########################################
# Setup
########################################

# Configure logging if desired
logging.basicConfig(level=logging.INFO)

# Initialize the LLM model for all nodes
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
        # Strip the first line if it's a fence, and the last if it’s a fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def merge_commands(existing: dict, new: dict) -> dict:
    """
    Merge two command objects that have the same "name".
    Recursively merges subcommands and flags, while reconciling descriptions.
    """
    merged = {"name": existing["name"]}

    # Merge descriptions: if they differ, combine them or pick the non-empty
    desc1 = existing.get("description", "")
    desc2 = new.get("description", "")
    if desc1 == desc2:
        merged["description"] = desc1
    else:
        if not desc1:
            merged["description"] = desc2
        elif not desc2:
            merged["description"] = desc1
        else:
            # Combine both distinct descriptions, preserving uniqueness
            merged["description"] = list({desc1, desc2})

    for key in ["subcommands", "flags"]:
        items_existing = existing.get(key, [])
        items_new = new.get(key, [])
        merged_items = items_existing.copy()
        for new_item in items_new:
            found = False
            for idx, exist_item in enumerate(merged_items):
                if exist_item.get("name") == new_item.get("name"):
                    merged_items[idx] = merge_commands(exist_item, new_item)
                    found = True
                    break
            if not found:
                merged_items.append(new_item)
        if merged_items:
            merged[key] = merged_items

    return merged

def update_aggregated_commands(agg: List[dict], new_commands: List[dict]) -> List[dict]:
    """
    Merge newly discovered commands into the aggregator.
    """
    for new_cmd in new_commands:
        found = False
        for i, existing_cmd in enumerate(agg):
            if existing_cmd.get("name") == new_cmd.get("name"):
                agg[i] = merge_commands(existing_cmd, new_cmd)
                found = True
                break
        if not found:
            agg.append(new_cmd)
    return agg

########################################
# State Definition
########################################

class ToolDiscoveryState(TypedDict):
    """
    The shared state for discovering a CLI tool's commands.
    """
    # The root path of the CLI tool (like "bin/kubectl-operator")
    tool_path: str
    # A queue of subcommand paths left to process (each item is a list of strings)
    subcommand_queue: List[List[str]]
    # Already processed subcommand paths
    processed_subcommands: List[List[str]]
    # The aggregator of discovered commands
    aggregated_commands: List[dict]
    # Flag to mark completion
    done: bool

########################################
# Graph Nodes
########################################

def init_state_node(state: ToolDiscoveryState) -> dict:
    """
    Initialize the subcommand queue with just the empty list (root command).
    """
    logging.info("Initializing tool discovery state.")
    return {
        "subcommand_queue": [[]],  # Start with root (no subcommand)
        "processed_subcommands": [],
        "aggregated_commands": [],
        "done": False,
    }

def call_tool_help_node(state: ToolDiscoveryState) -> dict:
    """
    Take the first subcommand path from the queue,
    call `tool_path <subcommand_path> help` via subprocess,
    store the output in the state.
    """
    new_state = copy.deepcopy(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state

    sub_path = queue[0]
    logging.info(f"Processing subcommand path: {sub_path}")

    # Build the command list, e.g. ["bin/kubectl-operator", "cluster", "install", "help"]
    cmd_list = [new_state["tool_path"]] + sub_path + ["help"]
    logging.info(f"Running command: {' '.join(cmd_list)}")

    try:
        completed_proc = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
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
    Parse the help text via the LLM to extract commands, subcommands, flags.
    Update aggregator and discover new subcommands to enqueue.
    """
    new_state = copy.deepcopy(state)
    help_text = new_state.get("help_output", "")
    sub_path = new_state["subcommand_queue"][0] if new_state["subcommand_queue"] else []

    prompt = f"""You are an expert at parsing CLI help text. Below is the full help output of a CLI tool for the subcommand path "{' '.join(sub_path)}". 
Extract a JSON structure with:
- "name": The command name
- "description": Brief description
- Optional "subcommands": list of similar objects
- Optional "flags": list of objects with "name" and "description"
Return a valid JSON object with a top-level "commands" list.

Help Output:
Return only the JSON, nothing else.
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

    # Merge into aggregator
    agg = new_state["aggregated_commands"]
    new_state["aggregated_commands"] = update_aggregated_commands(agg, commands_found)

    logging.info("Aggregated commands after parsing:")
    logging.info(json.dumps(new_state["aggregated_commands"], indent=2))

    # Gather subcommands found; queue them
    def collect_subcommands(cmd_list: List[dict], current_path: List[str]) -> List[List[str]]:
        new_paths = []
        for c in cmd_list:
            name = c.get("name")
            subcmds = c.get("subcommands", [])
            if name:
                new_paths.append(current_path + [name])
            if subcmds:
                deeper_paths = collect_subcommands(subcmds, current_path + [name])
                new_paths.extend(deeper_paths)
        return new_paths

    newly_discovered_paths = collect_subcommands(commands_found, sub_path)

    queue = new_state["subcommand_queue"]
    processed = new_state["processed_subcommands"]
    for p in newly_discovered_paths:
        if p not in queue and p not in processed:
            queue.append(p)

    return new_state

def mark_subcommand_processed_node(state: ToolDiscoveryState) -> dict:
    """
    Remove the front subcommand path from the queue and put it into processed_subcommands.
    """
    new_state = copy.deepcopy(state)
    if new_state["subcommand_queue"]:
        processed_sub = new_state["subcommand_queue"].pop(0)
        new_state["processed_subcommands"].append(processed_sub)
    return new_state

def check_done_node(state: ToolDiscoveryState) -> dict:
    """
    If the queue is empty, set done=True, else False.
    """
    if not state["subcommand_queue"]:
        state["done"] = True
    else:
        state["done"] = False
    return state

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

    # Workflow:
    # 1) init -> call_tool_help
    # 2) call_tool_help -> parse_help
    # 3) parse_help -> mark_processed
    # 4) mark_processed -> check_done
    # 5) check_done -> either done or back to call_tool_help

    g.set_entry_point("init")
    g.add_edge("init", "call_tool_help")
    g.add_edge("call_tool_help", "parse_help")
    g.add_edge("parse_help", "mark_processed")
    g.add_edge("mark_processed", "check_done")

    def condition(state: ToolDiscoveryState) -> str:
        return "finish" if state.get("done") else "continue"

    g.add_conditional_edges("check_done", condition, {
        "finish": END,
        "continue": "call_tool_help"
    })

    return g.compile()

########################################
# Main
########################################

if __name__ == "__main__":
    # Start with a single tool: "bin/kubectl-operator"
    initial_state: ToolDiscoveryState = {
        "tool_path": "bin/kubectl-operator",
        "subcommand_queue": [],
        "processed_subcommands": [],
        "aggregated_commands": [],
        "done": False,
    }

    graph = build_graph()

    try:
        result = graph.invoke(initial_state)
        # Wrap the aggregator in the final JSON with tool path at the top
        final_output = {
            "tool": result["tool_path"],
            "commands": result.get("aggregated_commands", [])
        }
        print("=== Final Discovered CLI Structure ===")
        print(json.dumps(final_output, indent=2))
    except Exception as e:
        logging.error(f"Graph execution error: {e}")