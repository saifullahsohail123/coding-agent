import os
import subprocess
import base64
import sqlite3
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
# from langchain_google_genai import ChatGoogleGenerativeAI # For future Gemini shift
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

# --- 1. PROJECT MANAGEMENT & PERSISTENCE ---

def select_project():
    """Interactive CLI to choose or create a project folder."""
    base_prefix = "project-"
    # List existing project folders
    try:
        existing_dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and d.startswith(base_prefix)])
    except:
        existing_dirs = []
    
    print("\n[Project Manager]")
    print("1. Create New Project")
    if existing_dirs:
        print("2. Open Existing Project")
    
    choice = input("\nSelect an option: ").strip()
    
    if choice == "1":
        name = input("Enter project name (e.g., 'fibo-script'): ").strip().replace(" ", "_")
        project_dir = f"{base_prefix}{name}"
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            print(f"[*] Created new project: {project_dir}")
        else:
            print(f"[*] Opening existing project: {project_dir}")
        return project_dir
    
    elif choice == "2" and existing_dirs:
        print("\nExisting Projects:")
        for idx, d in enumerate(existing_dirs):
            print(f"{idx + 1}. {d}")
        
        try:
            dir_idx = int(input("\nSelect project number: ")) - 1
            return existing_dirs[dir_idx]
        except:
            print("Invalid index. Defaulting to first project.")
            return existing_dirs[0]
    
    else:
        default_dir = f"{base_prefix}default"
        if not os.path.exists(default_dir): os.makedirs(default_dir)
        return default_dir

# --- GLOBAL CONFIG ---
PROJECT_ROOT = None
DB_PATH = None
memory = None

def initialize_project_env():
    global PROJECT_ROOT, DB_PATH, memory
    PROJECT_ROOT = select_project()
    DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")
    db_connection = sqlite3.connect(DB_PATH, check_same_thread=False)
    memory = SqliteSaver(db_connection)
    print(f"[*] Session Active in {PROJECT_ROOT}")

# MODEL
llm = ChatOllama(model="llama3.2-vision", temperature=0)

# --- 2. STATE ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The history"]
    files: List[str]
    images: List[str]
    iteration_count: int

# --- 3. TOOLS ---
@tool
def execute_shell(command: str):
    """Executes a terminal command inside the project root."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, cwd=PROJECT_ROOT)
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\nEXIT_CODE: {result.returncode}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def write_to_disk(filename: str, content: str):
    """Writes a file inside the project root."""
    with open(os.path.join(PROJECT_ROOT, filename), "w") as f:
        f.write(content)
    return f"File '{filename}' written."

@tool
def read_from_disk(filename: str):
    """Reads a file from the project root."""
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path): return "Error: Not found."
    with open(path, "r") as f: return f.read()

tools = [execute_shell, write_to_disk, read_from_disk]
llm_with_tools = llm.bind_tools(tools)

# --- 4. NODES ---
def planner_node(state: AgentState):
    sys_msg = SystemMessage(content="You are Antigravity-Lite. Use tools to solve tasks. Analyze images if provided.")
    return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}

def tool_executor_node(state: AgentState):
    last_msg = state['messages'][-1]
    tool_results = []
    for tool_call in last_msg.tool_calls:
        fn = {"execute_shell": execute_shell, "write_to_disk": write_to_disk, "read_from_disk": read_from_disk}[tool_call["name"]]
        tool_results.append(ToolMessage(content=str(fn.invoke(tool_call["args"])), tool_call_id=tool_call["id"]))
    return {"messages": tool_results, "iteration_count": state.get('iteration_count', 0) + 1}

def human_gate_node(state: AgentState):
    print("\n[!] PAUSED: Waiting for human input. Type your feedback or 'resume'.")
    return state

# --- 5. ROUTING ---
def router(state: AgentState):
    last_msg = state['messages'][-1]
    if not last_msg.tool_calls: return END
    err = any("EXIT_CODE: 0" not in str(m.content) for m in state['messages'] if isinstance(m, ToolMessage) and "EXIT_CODE" in str(m.content))
    if err and state['iteration_count'] < 5: return "planner"
    return "human_gate"

# --- 6. GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("human_gate", human_gate_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "tool_executor")
builder.add_conditional_edges("tool_executor", router, {"planner": "planner", "human_gate": "human_gate", END: END})
builder.add_edge("human_gate", "planner")

def get_app(checkpointer): return builder.compile(checkpointer=checkpointer, interrupt_before=["human_gate"])

# --- 7. RUN ---
def run_session(app, user_text: str, image_path: str = None):
    content = [{"type": "text", "text": user_text}]
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    for event in app.stream({"messages": [HumanMessage(content=content)], "iteration_count": 0}, {"configurable": {"thread_id": "main"}}, stream_mode="values"):
        if "messages" in event:
            msg = event["messages"][-1]
            if isinstance(msg, AIMessage) and msg.content: print(f"\n[Agent]: {msg.content}")

if __name__ == "__main__":
    initialize_project_env()
    app = get_app(memory)
    while True:
        inp = input("\n[You]: ").strip()
        if inp.lower() in ["exit", "quit"]: break
        img = None
        if inp.startswith("image:"):
            parts = inp.split(" ", 1)
            img = parts[0].replace("image:", "")
            inp = parts[1] if len(parts) > 1 else "Analyze"
        run_session(app, inp, image_path=img)
