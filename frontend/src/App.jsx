import React, { useState, useEffect, useRef } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [logs, setLogs] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isPaused, setIsPaused] = useState(false);
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState('');
  const [showProjectModal, setShowProjectModal] = useState(true);
  
  const ws = useRef(null);
  const logEndRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchProjects();
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs, messages]);

  const fetchProjects = async () => {
    const res = await fetch('http://localhost:8000/projects');
    const data = await res.json();
    setProjects(data.projects);
  };

  const handleSelectProject = async (name) => {
    await fetch('http://localhost:8000/projects/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    setSelectedProject(name);
    setShowProjectModal(false);
    connectWebSocket();
  };

  const connectWebSocket = () => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'agent_message') {
        if (data.role === 'assistant') {
          setMessages(prev => [...prev, { role: 'assistant', text: data.content }]);
        }
        if (data.tool_calls) {
          data.tool_calls.forEach(tc => {
            setLogs(prev => [...prev, { type: 'tool', text: `[*] Calling ${tc.name}(${JSON.stringify(tc.args)})` }]);
          });
        }
      } else if (data.type === 'status') {
        if (data.status === 'paused') {
          setIsPaused(true);
          setLogs(prev => [...prev, { type: 'status', text: '[!] Paused for approval' }]);
        } else {
          setIsPaused(false);
        }
      } else if (data.type === 'error') {
          setLogs(prev => [...prev, { type: 'error', text: `ERROR: ${data.message}` }]);
      }
    };
  };

  const sendMessage = (resumeText = null) => {
    if (!inputText && !resumeText) return;
    
    const payload = resumeText 
      ? { resume: true, text: resumeText }
      : { text: inputText };

    if (!resumeText) {
      setMessages(prev => [...prev, { role: 'user', text: inputText }]);
    }
    
    ws.current.send(JSON.stringify(payload));
    setInputText('');
    if (resumeText) setIsPaused(false);
  };

  return (
    <div className="app-container">
      {/* Project Selector Modal */}
      {showProjectModal && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>Select Project</h2>
            <div className="project-list">
              {projects.map(p => (
                <button key={p} onClick={() => handleSelectProject(p)}>{p}</button>
              ))}
              <button className="new" onClick={() => handleSelectProject(`project-${Date.now()}`)}>+ New Project</button>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar / Chat History */}
      <div className="sidebar">
        <div className="sidebar-header">
          <div className="logo">Antigravity Terminal</div>
          <div className="status-dot green"></div>
        </div>
        
        <div className="chat-history">
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              {m.text}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <textarea 
            placeholder="Type your instruction..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={3}
          />
          <button className="btn-send" onClick={() => sendMessage()}>Send Instruction</button>
        </div>
      </div>

      {/* Main Terminal View */}
      <div className="terminal-view">
        <div className="terminal-header">
          <div className="dot red"></div>
          <div className="dot yellow"></div>
          <div className="dot green"></div>
          <span style={{marginLeft: '10px', fontSize: '0.7rem', opacity: 0.5}}>{selectedProject}</span>
        </div>

        <div className="terminal-output">
          {logs.map((log, i) => (
            <div key={i} className={`log-entry ${log.type}`}>
              {log.text}
            </div>
          ))}
          <div ref={logEndRef} />
        </div>

        {/* HITL UI */}
        {isPaused && (
          <div className="hitl-overlay">
            <div className="hitl-header">Action Approval Required</div>
            <div className="hitl-actions">
              <button className="btn-approve" onClick={() => sendMessage('resume')}>Approve & Resume</button>
              <button className="btn-edit" onClick={() => setInputText('fix it like this...')}>Provide Feedback</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
