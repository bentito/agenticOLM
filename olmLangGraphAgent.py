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
    cached: bool  # flag indicating if cache was loaded
    human_command: str  # natural language query from the human
    tool_call: dict  # JSON output from the THINK step
    act_output: str  # output from executing the tool call
    quit: bool  # flag to exit human interaction loop


########################################
# Graph Nodes
########################################

def init_state_node(state: ToolDiscoveryState) -> dict:
    logging.info("Initializing tool discovery state.")
    state["cached"] = False
    state["quit"] = False
    return {
        "subcommand_queue": [[]],
        "processed_subcommands": [],
        "root_command": state["root_command"],
        "done": False,
        "cached": state["cached"],
        "quit": state["quit"]
    }


def cache_check_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    cache_dir = "tool_info"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        logging.info(f"Cache directory {cache_dir} created.")
    cache_filename = new_state["tool_path"].replace(os.sep, "_") + "_help.json"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    logging.info(f"Computed cache_filepath: {cache_filepath}")
    if os.path.exists(cache_filepath):
        logging.info(f"Cache found in cache_check_node for {new_state['tool_path']} at {cache_filepath}")
        with open(cache_filepath, "r") as f:
            cached_data = json.load(f)
        if "commands" in cached_data and cached_data["commands"]:
            new_state["root_command"] = cached_data["commands"][0]
        new_state["done"] = True
        new_state["cached"] = True
    else:
        logging.info(f"No cache found at {cache_filepath}, proceeding with discovery.")
    return new_state


def call_tool_help_node(state: ToolDiscoveryState) -> dict:
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
    new_state = copy.deepcopy(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state
    sub_path = queue[0]
    help_text = new_state.get("help_output", "")
    root_cmd = new_state["root_command"]
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
        logging.info("No subcommands or flags discovered.")
        return new_state
    current_cmd_data = commands_found[0]
    if len(sub_path) == 0:
        desc = current_cmd_data.get("description", "")
        if desc:
            root_cmd["description"] = desc
        new_flags = current_cmd_data.get("flags", [])
        old_flags = root_cmd.get("flags", [])
        root_cmd["flags"] = merge_flags(old_flags, new_flags)
        if "subcommands" in current_cmd_data:
            root_cmd["subcommands"] = current_cmd_data["subcommands"]
    else:
        parent_path = sub_path[:-1]
        this_sub_name = sub_path[-1]
        parent_cmd = ensure_command_in_path(root_cmd, parent_path)
        sub_list = parent_cmd.get("subcommands", [])
        existing_sub = None
        for sc in sub_list:
            if sc["name"] == this_sub_name:
                existing_sub = sc
                break
        if not existing_sub:
            existing_sub = {"name": this_sub_name, "description": "", "subcommands": []}
            sub_list.append(existing_sub)
        old_desc = existing_sub.get("description", "")
        new_desc = current_cmd_data.get("description", "")
        if new_desc and (new_desc != old_desc):
            if not old_desc:
                existing_sub["description"] = new_desc
            elif new_desc not in old_desc:
                existing_sub["description"] = old_desc + " / " + new_desc
        old_flags = existing_sub.get("flags", [])
        found_flags = current_cmd_data.get("flags", [])
        existing_sub["flags"] = merge_flags(old_flags, found_flags)
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
        if p not in new_state["subcommand_queue"] and p not in processed and p != sub_path:
            if len(p) >= 2 and p[-1] == p[-2]:
                logging.warning(f"Skipping potential self-cycle path: {p}")
                continue
            new_state["subcommand_queue"].append(p)
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


# Unused trial node remains.
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


# New node: Human interaction.
def human_interaction_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    user_input = input("How can I help you manage operators or cluster extensions? (or 'quit' to exit): ")
    if user_input.strip().lower() == "quit":
        new_state["quit"] = True
    else:
        new_state["human_command"] = user_input.strip()
    return new_state


# New node: THINK step - process human natural language into a structured tool call.
def think_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    if "human_command" not in new_state or not new_state["human_command"]:
        logging.warning("No human command provided to THINK step.")
        return new_state
    # Provide the full JSON summary of the tool.
    tool_summary = json.dumps(new_state["root_command"], indent=2)
    prompt = f"""You are an expert at formulating tool calls for managing operators or cluster extensions.
Below is the JSON summary of the tool, which describes all available commands and flags:
{tool_summary}

The human query is:
{new_state['human_command']}

Your task: Based on the JSON summary, generate a tool call that uses one of the valid commands from the summary to answer the human query. 
Return only a valid JSON object representing the tool call with keys "tool", "function", and "arguments". 
Do not include any extra commentary.
"""
    logging.info("THINK node prompt generated (full JSON summary provided).")
    think_llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, verbose=True)
    response = think_llm.invoke([HumanMessage(content=prompt)])
    content = clean_markdown(response.content)
    logging.info("THINK node LLM response:")
    logging.info(content)
    try:
        new_state["tool_call"] = json.loads(content)
    except Exception as e:
        logging.warning(f"Failed to parse THINK node tool call JSON: {e}")
        new_state["tool_call"] = None
    return new_state


# Modified ACT node: execute the structured tool call.
def act_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    if "tool_call" not in new_state or not new_state["tool_call"]:
        new_state["act_output"] = "No valid tool call generated."
        return new_state
    tool_call = new_state["tool_call"]
    function = tool_call.get("function", "")
    arguments = tool_call.get("arguments", [])
    # If arguments is a dict, flatten it into a list of command-line arguments.
    if isinstance(arguments, dict):
        flat_args = []
        for k, v in arguments.items():
            flat_args.append(str(k))
            if not (isinstance(v, bool) and v is True):
                flat_args.append(str(v))
        arguments = flat_args
    elif not isinstance(arguments, list):
        arguments = [str(arguments)]
    cmd_list = [new_state["tool_path"], function] + arguments
    logging.info(f"Executing command: {' '.join(cmd_list)}")
    try:
        proc = subprocess.run(cmd_list, capture_output=True, text=True, check=False)
        new_state["act_output"] = proc.stdout + "\n" + proc.stderr
    except Exception as e:
        new_state["act_output"] = f"Error: {str(e)}"
    return new_state


# New node: Observe the result.
def observe_node(state: ToolDiscoveryState) -> dict:
    logging.info("Observation: " + state.get("act_output", ""))
    return state


# New node: Write the final result to cache.
def cache_write_node(state: ToolDiscoveryState) -> dict:
    new_state = copy.deepcopy(state)
    cache_dir = "tool_info"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_filename = new_state["tool_path"].replace(os.sep, "_") + "_help.json"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    final_output = {
        "tool_path": new_state["tool_path"],
        "commands": [new_state["root_command"]],
        "tool_call": new_state.get("tool_call")
    }
    logging.info(f"Writing cache to {cache_filepath}")
    with open(cache_filepath, "w") as f:
        json.dump(final_output, f, indent=2)
    if not new_state.get("cached", False):
        logging.info("=== Final Discovered CLI Structure ===")
        logging.info(json.dumps(final_output, indent=2))
    new_state["final_output"] = final_output
    return new_state


########################################
# Build the Graph
########################################

def build_graph() -> StateGraph:
    g = StateGraph(ToolDiscoveryState)
    g.add_node("init", init_state_node)
    g.add_node("cache_check", cache_check_node)
    g.add_node("call_tool_help", call_tool_help_node)
    g.add_node("parse_help", parse_tool_help_node)
    g.add_node("mark_processed", mark_subcommand_processed_node)
    g.add_node("check_done", check_done_node)
    g.add_node("trial_tool_call", trial_tool_call_node)  # Unused in current flow.
    g.add_node("cache_write", cache_write_node)
    g.add_node("human_interaction", human_interaction_node)
    g.add_node("think", think_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)

    g.set_entry_point("init")

    def cache_condition(state: ToolDiscoveryState) -> str:
        return "finish" if state.get("done") else "continue"

    g.add_edge("init", "cache_check")
    g.add_conditional_edges("cache_check", cache_condition, {
        "finish": "cache_write",
        "continue": "call_tool_help"
    })
    g.add_edge("call_tool_help", "parse_help")
    g.add_edge("parse_help", "mark_processed")

    def discovery_condition(state: ToolDiscoveryState) -> str:
        return "finish" if state.get("done") else "continue"

    g.add_conditional_edges("mark_processed", discovery_condition, {
        "finish": "cache_write",
        "continue": "check_done"
    })
    g.add_edge("check_done", "call_tool_help")
    g.add_edge("cache_write", "human_interaction")
    g.add_edge("human_interaction", "think")
    g.add_edge("think", "act")
    g.add_edge("act", "observe")

    def human_loop(state: ToolDiscoveryState) -> str:
        return "continue" if not state.get("quit", False) else "finish"

    g.add_conditional_edges("observe", human_loop, {
        "continue": "human_interaction",
        "finish": END
    })
    return g.compile()


########################################
# Main
########################################

if __name__ == "__main__":
    tools = ["bin/kubectl-operator"]
    graph = build_graph()
    results = []
    for tool_path in tools:
        tool_basename = os.path.basename(tool_path)
        initial_state: ToolDiscoveryState = {
            "tool_path": tool_path,
            "subcommand_queue": [],
            "processed_subcommands": [],
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
            results.append(result.get("final_output", {}))
        except Exception as e:
            logging.error(f"Graph execution error for {tool_path}: {e}")