import os
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import the agent logic
import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProjectInfo(BaseModel):
    name: str

@app.get("/projects")
async def list_projects():
    base_prefix = "project-"
    try:
        dirs = sorted([d for d in os.listdir(".") if os.path.isdir(d) and d.startswith(base_prefix)])
    except:
        dirs = []
    return {"projects": dirs}

@app.post("/projects/select")
async def select_project(project: ProjectInfo):
    # This simulates the logic in agent.select_project but driven by API
    project_dir = project.name
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    
    agent.PROJECT_ROOT = project_dir
    agent.DB_PATH = os.path.join(project_dir, "chat_history.db")
    
    # We use AsyncSqliteSaver for the web backend
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    agent.memory = AsyncSqliteSaver.from_conn_string(agent.DB_PATH)
    
    return {"status": "success", "project": project_dir}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if not agent.memory:
        await websocket.send_json({"type": "error", "message": "Project not selected"})
        await websocket.close()
        return
        
    # AsyncSqliteSaver.from_conn_string is a context manager
    async with agent.memory as saver:
        compiled_agent = agent.get_app(saver)
        
        try:
            while True:
                data = await websocket.receive_text()
                payload = json.loads(data)
                
                user_text = payload.get("text")
                image_b64 = payload.get("image")
                is_resume = payload.get("resume", False)
                
                config = {"configurable": {"thread_id": "main"}}
                
                if is_resume:
                    if user_text:
                        await compiled_agent.aupdate_state(config, {"messages": [HumanMessage(content=user_text)]})
                    stream_input = None
                else:
                    content = [{"type": "text", "text": user_text}]
                    if image_b64:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
                    stream_input = {"messages": [HumanMessage(content=content)], "iteration_count": 0}
                
                # Stream messages
                async for msg, metadata in compiled_agent.astream(stream_input, config, stream_mode="messages"):
                    msg_data = {
                        "type": "agent_message",
                        "role": "assistant" if isinstance(msg, AIMessage) else "tool",
                        "content": msg.content,
                        "metadata": metadata
                    }
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        msg_data["tool_calls"] = msg.tool_calls
                    
                    await websocket.send_json(msg_data)
                    
                # Check for breakpoints
                state = await compiled_agent.aget_state(config)
                if state.next:
                    await websocket.send_json({"type": "status", "status": "paused", "message": "Action requires approval"})
                else:
                    await websocket.send_json({"type": "status", "status": "done"})

        except WebSocketDisconnect:
            print("Client disconnected")
        except Exception as e:
            print(f"Error: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
