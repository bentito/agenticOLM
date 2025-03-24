import requests
import logging
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict
import operator

# Set up logging.
logging.basicConfig(level=logging.INFO)


# Define the state for our repository discovery workflow.
class RepoDiscoveryState(TypedDict):
    repo_url: str
    file_list: list[str]
    selected_file: str
    file_content: str
    extracted_commands: str
    iteration: int  # Track recursion depth


# Helper: Parse owner and repo from a GitHub URL.
def get_repo_owner_and_name(repo_url: str) -> (str, str):
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    else:
        raise Exception("Invalid repo URL format")


# --- Node 1: Discover candidate files using the GitHub API ---
def discover_repo(state: RepoDiscoveryState) -> dict:
    repo_url = state["repo_url"]
    owner, repo = get_repo_owner_and_name(repo_url)
    branch = "main"  # Adjust if necessary.
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    logging.info("Fetching repository file list from: %s", api_url)
    response = requests.get(api_url)
    if response.status_code != 200:
        logging.error("Error fetching repo files: %s", response.status_code)
        candidate_files = []
    else:
        tree = response.json().get("tree", [])
        candidate_files = [
            item["path"]
            for item in tree
            if item["type"] == "blob" and item["path"].endswith((".go", ".md", ".txt"))
        ]
    print("Discovered candidate files:", candidate_files)
    return {"file_list": candidate_files}


# Initialize the LLM model.
llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=False)


# --- Node 2: Select a file to analyze using GPT-4o ---
def select_file(state: RepoDiscoveryState) -> dict:
    file_list = state["file_list"]
    prompt = f"""You are an expert in CLI tool analysis.
Given the following list of files from a repository:
{file_list}
Which file is most likely to contain the definitions for CLI commands (entry points, flags, subcommands)?
Respond with only the file name.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    selected = response.content.strip()
    print("Selected file:", selected)
    return {"selected_file": selected}


# --- Node 3: Fetch the content of the selected file ---
def fetch_file_content(state: RepoDiscoveryState) -> dict:
    selected = state["selected_file"]
    owner, repo = get_repo_owner_and_name(state["repo_url"])
    branch = "main"  # Adjust if needed.
    base_raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/"
    file_url = base_raw_url + selected
    response = requests.get(file_url)
    if response.status_code == 200:
        content = response.text
        print(f"Fetched content from {selected} (length {len(content)} characters)")
    else:
        content = f"Error: failed to fetch {selected}"
        print(content)
    return {"file_content": content}


# --- Node 4: Extract CLI commands from the file content ---
def extract_commands(state: RepoDiscoveryState) -> dict:
    content = state["file_content"]
    prompt = f"""You are a tool extraction assistant with expertise in programming languages.
Analyze the following file content from a CLI tool's codebase and extract all commands, subcommands,
flags, and options. Return a valid JSON object with a single key "commands", whose value is a list of objects.
Each object should have:
  - "name": the command or subcommand name,
  - "description": a brief description of what the command does.
Return only the JSON output without additional commentary.

File Content:
{content}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    extracted = response.content.strip()
    print("Extracted commands JSON:", extracted)
    return {"extracted_commands": extracted}


# --- Node 5: Check extraction and decide whether to continue ---
def check_extraction(state: RepoDiscoveryState) -> dict:
    iteration = state.get("iteration", 0)
    extracted = state["extracted_commands"]
    print(f"Iteration {iteration}: Extracted commands length: {len(extracted)}")
    # If extraction appears too minimal (less than 100 chars), continue; else, finish.
    if len(extracted) < 100 and iteration < 5:
        print("Extraction too minimal; need to revisit another file.")
        current = state["selected_file"]
        candidates = state["file_list"]
        remaining = [f for f in candidates if f != current]
        if remaining:
            new_selection = remaining[0]
            print("New selected file:", new_selection)
            return {
                "selected_file": new_selection,
                "extracted_commands": "",
                "iteration": iteration + 1
            }
        else:
            print("No more candidate files available; stopping recursion.")
            return {"extracted_commands": extracted}
    else:
        # If extraction is sufficient, signal that we're done.
        return {"extracted_commands": extracted, "done": True}


# --- Build the state graph integrating nodes 1 to 5 with conditional edge ---
def build_graph() -> StateGraph:
    graph_builder = StateGraph(RepoDiscoveryState)
    graph_builder.add_node("discover", discover_repo)
    graph_builder.add_node("select", select_file)
    graph_builder.add_node("fetch", fetch_file_content)
    graph_builder.add_node("extract", extract_commands)
    graph_builder.add_node("check", check_extraction)

    # Define the workflow:
    # discover -> select -> fetch -> extract -> check
    graph_builder.set_entry_point("discover")
    graph_builder.add_edge("discover", "select")
    graph_builder.add_edge("select", "fetch")
    graph_builder.add_edge("fetch", "extract")
    graph_builder.add_edge("extract", "check")

    # Conditional edge: from check, if state contains "done" == True, go to END; else loop back to fetch.
    def check_condition(state: RepoDiscoveryState) -> str:
        return "finish" if state.get("done", False) else "continue"

    graph_builder.add_conditional_edges("check", check_condition, {"finish": END, "continue": "fetch"})
    return graph_builder.compile()


# --- Main execution ---
if __name__ == "__main__":
    # Initialize the state.
    initial_state: RepoDiscoveryState = {
        "repo_url": "https://github.com/operator-framework/kubectl-operator",
        "file_list": [],
        "selected_file": "",
        "file_content": "",
        "extracted_commands": "",
        "iteration": 0
    }
    graph = build_graph()
    try:
        result = graph.invoke(initial_state)
        print("=== Final Result ===")
        print(result)
    except Exception as e:
        print("Graph execution error:", str(e))