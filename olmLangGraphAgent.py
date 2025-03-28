import os

# Force Python to run unbuffered (like python -u)
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
import json
import logging
import copy
import subprocess
from typing import List, Dict, Any
from typing_extensions import TypedDict

# If your environment re-imports or resets logging, you might do it AFTER
# setting PYTHONUNBUFFERED. We'll reconfigure logging to use stdout explicitly:
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers:
    logger.removeHandler(h)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

# Now do the rest of your imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

########################################
# Setup
########################################

# The main LLM used for both discovery and user queries.
llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=False)

########################################
# Tools and their specialized help calls
########################################

TOOL_PATHS = ["bin/kubectl-operator", "/usr/bin/grep"]

HELP_CALL_MAP = {
    "bin/kubectl-operator": "subcommand",
    "/usr/bin/grep": "flag"
}


########################################
# Flush Utility
########################################

def flush_all():
    """
    Flush stdout and all log handlers.
    This helps ensure messages appear immediately in most terminals.
    """
    sys.stdout.flush()
    # Also flush any logging handler. We set all to stdout above, so this is mostly redundant.
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except:
            pass


########################################
# Helpers
########################################

def clean_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def merge_flags(existing_flags: List[dict], new_flags: List[dict]) -> List[dict]:
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


def run_help_command(tool_path: str, sub_path: List[str]) -> (str, int):
    style = HELP_CALL_MAP.get(tool_path, "subcommand")
    if style == "subcommand":
        cmd_list = [tool_path] + sub_path + ["help"]
    else:
        cmd_list = [tool_path, "--help"]

    logging.info(f"Running command: {' '.join(cmd_list)}")
    try:
        completed_proc = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        return (completed_proc.stdout + "\n" + completed_proc.stderr, completed_proc.returncode)
    except subprocess.TimeoutExpired as te:
        logging.error(f"Tool help command timed out: {cmd_list}")
        return (f"Timed out after 10s: {cmd_list}", 1)
    except Exception as e:
        logging.error(f"Error running {tool_path}: {e}")
        return (f"Error: {str(e)}", 1)


def parse_help_text_with_llm(sub_path: List[str], help_text: str) -> dict:
    prompt = f"""You are an expert at extracting CLI structure from help text. 
We have a command path: {' '.join(sub_path)}.

Your job:
1. Read the help text below carefully for lines about usage, available commands, flags.
2. Identify subcommands and flags.
3. Return a JSON with a top-level "commands" array containing exactly one object:
   {{
     "name": "<the command name for this path>",
     "description": "<short desc>",
     "subcommands": [ ... ],
     "flags": [ {{ "name": "...", "description": "..." }} ]
   }}
Help Output:
{help_text}
Return only the valid JSON.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = clean_markdown(response.content)
    try:
        return json.loads(content)
    except:
        logging.warning("Failed to parse JSON from LLM parse output.")
        return {}


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


def integrate_parsed_help(
        root_cmd: dict, sub_path: List[str], parsed_help: dict, processed: List[List[str]]
) -> None:
    commands_found = parsed_help.get("commands", [])
    if not commands_found:
        logging.info("No commands found in parse.")
        return

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
        if new_desc and new_desc not in old_desc:
            if not old_desc:
                existing_sub["description"] = new_desc
            else:
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
    newly_discovered = collect_subcommands(current_cmd_data, sub_path)
    for nd in newly_discovered:
        if nd not in processed and nd != sub_path:
            processed.append(nd)


########################################
# State Definition
########################################

class MasterState(TypedDict, total=False):
    tools: List[str]
    discovered_structs: Dict[str, Any]
    current_tool_index: int
    subcommand_queue: List[List[str]]
    processed_subcommands: List[List[str]]
    done_discovery: bool
    user_query: str
    quit: bool
    final_output: Dict[str, Any]
    help_output: str
    return_code: int
    tool_sequence: List[Dict[str, Any]]
    tool_output: str


########################################
# Graph Nodes
########################################

def init_state_node(state: MasterState) -> Dict[str, Any]:
    logging.info("Init node: Setting up for tool discovery.")
    flush_all()
    new_state = dict(state)
    new_state["discovered_structs"] = {}
    new_state["current_tool_index"] = 0
    new_state["subcommand_queue"] = [[]]
    new_state["processed_subcommands"] = []
    new_state["done_discovery"] = False
    new_state["quit"] = False
    new_state["help_output"] = ""
    new_state["return_code"] = 0
    new_state["tool_sequence"] = []
    new_state["tool_output"] = ""
    return new_state


def check_cache_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    tool_path = new_state["tools"][new_state["current_tool_index"]]
    cache_dir = "tool_info"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_filename = tool_path.replace(os.sep, "_") + "_help.json"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_filepath):
        logging.info(f"Cache found for {tool_path} at {cache_filepath}, loading.")
        with open(cache_filepath, "r") as f:
            cached_data = json.load(f)
        if "commands" in cached_data and len(cached_data["commands"]) > 0:
            new_state["discovered_structs"][tool_path] = cached_data["commands"][0]
        new_state["subcommand_queue"] = []
        new_state["processed_subcommands"] = []
    else:
        tool_basename = os.path.basename(tool_path)
        root_cmd = {
            "name": tool_basename,
            "description": "",
            "subcommands": [],
            "flags": []
        }
        new_state["discovered_structs"][tool_path] = root_cmd
        new_state["subcommand_queue"] = [[]]
        new_state["processed_subcommands"] = []
    return new_state


def call_tool_help_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state
    sub_path = queue[0]
    tool_path = new_state["tools"][new_state["current_tool_index"]]
    help_output, rc = run_help_command(tool_path, sub_path)
    new_state["help_output"] = help_output
    new_state["return_code"] = rc
    return new_state


def parse_tool_help_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    queue = new_state["subcommand_queue"]
    if not queue:
        return new_state
    sub_path = queue[0]
    help_output = new_state.get("help_output", "")
    tool_path = new_state["tools"][new_state["current_tool_index"]]
    root_command = new_state["discovered_structs"][tool_path]
    parsed_help = parse_help_text_with_llm(sub_path, help_output)
    integrate_parsed_help(root_command, sub_path, parsed_help, new_state["processed_subcommands"])
    return new_state


def mark_subcommand_processed_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    if new_state["subcommand_queue"]:
        new_state["subcommand_queue"].pop(0)
    return new_state


def queue_new_subs_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    processed = new_state["processed_subcommands"]
    queued = set(tuple(x) for x in new_state["subcommand_queue"])
    to_add = []
    for p in processed:
        tp = tuple(p)
        if tp not in queued:
            to_add.append(p)
    new_state["subcommand_queue"].extend(to_add)
    return new_state


def check_done_tool_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    if len(new_state["subcommand_queue"]) == 0:
        tool_path = new_state["tools"][new_state["current_tool_index"]]
        cache_dir = "tool_info"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_filename = tool_path.replace(os.sep, "_") + "_help.json"
        cache_filepath = os.path.join(cache_dir, cache_filename)
        final_output = {
            "tool_path": tool_path,
            "commands": [new_state["discovered_structs"][tool_path]],
        }
        with open(cache_filepath, "w") as f:
            json.dump(final_output, f, indent=2)
        logging.info(f"Discovery for {tool_path} completed, wrote cache to {cache_filepath}")
        new_state["current_tool_index"] += 1
        if new_state["current_tool_index"] >= len(new_state["tools"]):
            new_state["done_discovery"] = True
        else:
            new_state["subcommand_queue"] = [[]]
            new_state["processed_subcommands"] = []
    return new_state


def check_all_tools_done_node(state: MasterState) -> Dict[str, Any]:
    return state


def user_interaction_node(state: MasterState) -> Dict[str, Any]:
    flush_all()
    new_state = dict(state)
    user_input = input("OLM related directives? ")
    new_state["user_query"] = user_input.strip()
    if user_input.strip().lower() == "quit":
        new_state["quit"] = True
    return new_state


def think_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    if new_state.get("quit"):
        return new_state
    if not new_state.get("user_query"):
        logging.warning("No user query found.")
        return new_state

    all_tools_summary = []
    for tpath, struct in new_state["discovered_structs"].items():
        all_tools_summary.append(struct)
    system_msg_text = (
            "You are a helpful CLI orchestrator. We have these tools discovered:\n\n"
            + json.dumps(all_tools_summary, indent=2)
            + "\n\n"
              "You may combine these tools. For example, you can run a command in 'kubectl-operator' "
              "and then pipe its output to 'grep'. Or just use 'grep' alone. Please produce a valid JSON "
              "structure with the top-level key `tool_sequence` (a list of steps). Each step is an object "
              "with `tool_path` and `arguments` to run that tool. For a pipe, just put multiple steps.\n\n"
              "The user request is: "
            + new_state["user_query"]
            + "\n\n"
              "Return ONLY the JSON, no extra text. The JSON format:\n"
              "{\n"
              "  \"tool_sequence\": [\n"
              "    {\n"
              "      \"tool_path\": \"...\",\n"
              "      \"arguments\": [\"...\",\"...\"]\n"
              "    },\n"
              "    ...\n"
              "  ]\n"
              "}"
    )
    logging.info("Invoking LLM with combined tool schema.")
    flush_all()
    response = llm.invoke([HumanMessage(content=system_msg_text)])
    content = clean_markdown(response.content)
    try:
        parsed = json.loads(content)
        new_state["tool_sequence"] = parsed.get("tool_sequence", [])
    except Exception as e:
        logging.warning(f"Failed to parse tool_sequence: {e}")
        new_state["tool_sequence"] = []
    return new_state


def act_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    seq = new_state.get("tool_sequence", [])
    if not seq:
        logging.info("No tool steps to execute.")
        flush_all()
        return new_state

    last_output = ""
    for step_idx, step in enumerate(seq):
        tool_path = step.get("tool_path")
        arguments = step.get("arguments", [])
        logging.info(f"Executing step {step_idx + 1}: {tool_path} {arguments}")
        flush_all()
        try:
            if "grep" in tool_path:
                proc = subprocess.Popen(
                    [tool_path] + arguments,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = proc.communicate(input=last_output)
                last_output = stdout + "\n" + stderr
            else:
                proc = subprocess.run(
                    [tool_path] + arguments,
                    capture_output=True,
                    text=True,
                    check=False
                )
                last_output = proc.stdout + "\n" + proc.stderr
        except Exception as e:
            last_output = f"Error: {str(e)}"
            logging.error("Error executing tool step: %s", e)
            flush_all()
            break
    new_state["tool_output"] = last_output
    return new_state


def observe_node(state: MasterState) -> Dict[str, Any]:
    logging.info("FINAL OUTPUT:")
    print(state.get("tool_output", "No output"), flush=True)
    flush_all()
    return state


def finish_node(state: MasterState) -> Dict[str, Any]:
    new_state = dict(state)
    new_state["final_output"] = "Session ended."
    return new_state


########################################
# Build the Graph
########################################

def build_graph() -> StateGraph:
    g = StateGraph(MasterState)

    g.add_node("init", init_state_node)
    g.add_node("check_cache", check_cache_node)
    g.add_node("call_help", call_tool_help_node)
    g.add_node("parse_help", parse_tool_help_node)
    g.add_node("mark_processed", mark_subcommand_processed_node)
    g.add_node("queue_new_subs", queue_new_subs_node)
    g.add_node("check_done_tool", check_done_tool_node)
    g.add_node("check_all_tools_done", check_all_tools_done_node)
    g.add_node("user_interact", user_interaction_node)
    g.add_node("think", think_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)
    g.add_node("finish", finish_node)

    g.set_entry_point("init")

    def all_tools_done_cond(state: MasterState) -> str:
        return "done" if state.get("done_discovery", False) else "continue"

    g.add_edge("init", "check_cache")

    def have_subcommands(state: MasterState) -> str:
        return "no" if len(state.get("subcommand_queue", [])) == 0 else "yes"

    g.add_conditional_edges("check_cache", have_subcommands, {
        "no": "check_done_tool",
        "yes": "call_help"
    })

    g.add_edge("call_help", "parse_help")
    g.add_edge("parse_help", "mark_processed")
    g.add_edge("mark_processed", "queue_new_subs")

    def queue_is_empty(state: MasterState) -> str:
        return "tool_done" if len(state.get("subcommand_queue", [])) == 0 else "need_more"

    g.add_conditional_edges("queue_new_subs", queue_is_empty, {
        "tool_done": "check_done_tool",
        "need_more": "call_help"
    })

    g.add_edge("check_done_tool", "check_all_tools_done")
    g.add_conditional_edges("check_all_tools_done", all_tools_done_cond, {
        "done": "user_interact",
        "continue": "check_cache"
    })

    def quit_cond(state: MasterState) -> str:
        if state.get("quit"):
            return "exit"
        if state.get("user_query"):
            return "think"
        return "user_interact"

    g.add_conditional_edges("user_interact", quit_cond, {
        "exit": "finish",
        "think": "think",
        "user_interact": "user_interact"
    })

    g.add_edge("think", "act")
    g.add_edge("act", "observe")

    def user_loop_cond(state: MasterState) -> str:
        if state.get("quit"):
            return "exit"
        return "user_interact"

    g.add_conditional_edges("observe", user_loop_cond, {
        "exit": "finish",
        "user_interact": "user_interact"
    })

    g.add_edge("finish", END)

    return g.compile()


########################################
# Main
########################################

if __name__ == "__main__":
    # If your shell/IDE doesn't respect PYTHONUNBUFFERED for some reason,
    # also run with python -u <script> or set PYTHONUNBUFFERED=1 externally.
    graph = build_graph()
    initial_state: MasterState = {
        "tools": TOOL_PATHS,
        "discovered_structs": {},
        "current_tool_index": 0,
        "done_discovery": False
    }
    config = {"recursion_limit": 200, "configurable": {"thread_id": "multi-tool-thread"}}
    result_state = graph.invoke(initial_state, config)
    logging.info("Graph execution finished.")
    flush_all()
    logging.info(f"Final State: {result_state}")