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

# MODELS
# Reasoning model (Supports tools)
llm_tools = ChatOllama(model="llama3.1", temperature=0)
# Vision model (Supports images, used only for description)
llm_vision = ChatOllama(model="llama3.2-vision", temperature=0)

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
llm_with_tools = llm_tools.bind_tools(tools)

# --- 4. NODES ---
def planner_node(state: AgentState):
    """Uses Hybrid approach: Vision model for images, Tool model for logic."""
    messages = state['messages']
    last_msg = messages[-1]
    
    # If the user sent an image, we use the vision model to describe it first 
    # so the tool model can 'understand' what's in the picture.
    processed_history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            # Check if this message has an image
            has_image = any(item.get("type") == "image_url" for item in msg.content if isinstance(item, dict))
            if has_image:
                print("[System] Detected image. Requesting visual analysis from llama3.2-vision...")
                vision_res = llm_vision.invoke([msg])
                # Convert the multimodal message into a purely text message for the tool model
                description = f"(Visual Content Analysis: {vision_res.content})"
                processed_history.append(HumanMessage(content=description))
                continue
        processed_history.append(msg)

    sys_msg = SystemMessage(content=(
        "You are 'Antigravity-lite'. Use tools to solve coding tasks. "
        "You have been provided with textual descriptions of any images uploaded."
    ))

    # Use the reasoning model (which supports tools) for the actual plan
    response = llm_with_tools.invoke([sys_msg] + processed_history)
    
    return {"messages": [response]}

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

# --- 5. ROUTING LOGIC ---

def route_after_planner(state: AgentState):
    """Decides if we should execute tools or end the session."""
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tool_executor"
    return END

def route_after_tools(state: AgentState):
    """Decides what to do after tools run: Fix errors or wait for human."""
    # Check for errors in the tool outputs
    error_detected = any("EXIT_CODE: 0" not in str(m.content) for m in state['messages'] if isinstance(m, ToolMessage) and "EXIT_CODE" in str(m.content))
    
    if error_detected and state['iteration_count'] < 5:
        print(f"[System] Error detected (Iteration {state['iteration_count']}). Returning to planner for fix...")
        return "planner"
    
    return "human_gate"

# --- 6. ASSEMBLE THE GRAPH ---

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tool_executor", tool_executor_node)
builder.add_node("human_gate", human_gate_node)

builder.set_entry_point("planner")

# From planner, we either go to tools or end
builder.add_conditional_edges("planner", route_after_planner, {
    "tool_executor": "tool_executor",
    END: END
})

# From tools, we either go back to planner (for fix) or to human gate
builder.add_conditional_edges("tool_executor", route_after_tools, {
    "planner": "planner",
    "human_gate": "human_gate"
})
# From human gate, we always go back to planner to process the feedback
builder.add_edge("human_gate", "planner")

def get_app(checkpointer): 
    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_gate"])

# --- 7. RUN ---
def run_session(app, user_text: str = None, image_path: str = None):
    # Determine the config (thread)
    config = {"configurable": {"thread_id": "main"}}
    
    # Check if we are resuming from an interrupt (human feedback)
    state = app.get_state(config)
    
    if state.next:
        # We are at a breakpoint (human_gate)
        if user_text:
            # Inject human feedback into the state
            app.update_state(config, {"messages": [HumanMessage(content=user_text)]})
        # Resume execution
        stream_input = None
    else:
        # Start a fresh interaction
        content = [{"type": "text", "text": user_text}]
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        stream_input = {"messages": [HumanMessage(content=content)], "iteration_count": 0}

    # Stream the graph
    for event in app.stream(stream_input, config, stream_mode="values"):
        if "messages" in event:
            msg = event["messages"][-1]
            if isinstance(msg, AIMessage):
                if msg.content: print(f"\n[Agent]: {msg.content}")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[*] Agent is calling tool: {tc['name']} with {tc['args']}")
            elif isinstance(msg, ToolMessage):
                # Print a summary of tool results
                status = "Success" if "EXIT_CODE: 0" in str(msg.content) or "Success" in str(msg.content) else "Error/Notice"
                print(f"[Tool Result]: {status}")

if __name__ == "__main__":
    initialize_project_env()
    app = get_app(memory)
    print("\n--- Antigravity-Lite Started ---")
    print("Type 'exit' to quit. Use 'image:/path/to/img.jpg <prompt>' for images.")
    
    while True:
        # Check if we are waiting for human feedback
        config = {"configurable": {"thread_id": "main"}}
        state = app.get_state(config)
        
        prompt = "[You]: "
        if state.next:
            prompt = "[Human Feedback required]: "
            
        inp = input(prompt).strip()
        if not inp: continue
        if inp.lower() in ["exit", "quit"]: break
        
        img = None
        if inp.startswith("image:"):
            parts = inp.split(" ", 1)
            img = parts[0].replace("image:", "")
            inp = parts[1] if len(parts) > 1 else "Analyze"
        
        run_session(app, inp, image_path=img)
