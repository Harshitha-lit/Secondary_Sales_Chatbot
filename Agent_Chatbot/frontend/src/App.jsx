import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

import './App.css';

const API_URL = `http://${window.location.hostname}:8001/chat`;

function TypewriterMarkdown({ content, onComplete }) {
  const [displayedContent, setDisplayedContent] = useState('');
  
  useEffect(() => {
    let i = 0;
    setDisplayedContent('');
    
    if (!content) {
      if (onComplete) onComplete();
      return;
    }
    
    const intervalId = setInterval(() => {
      setDisplayedContent(content.substring(0, i));
      i += 5; 
      
      if (i > content.length) {
        clearInterval(intervalId);
        setDisplayedContent(content); 
        if (onComplete) onComplete();
      }
    }, 20); 
    
    return () => clearInterval(intervalId);
  }, [content, onComplete]);

  return <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{displayedContent}</ReactMarkdown>;
}

function EvidenceCard({ evidence }) {
  if (!evidence || !evidence.sources || evidence.sources.length === 0) return null;
  const { sources, confidence, provenance } = evidence;
  
  const confPercent = Math.round((confidence || 0) * 100);
  
  let confColor = "#10B981"; 
  if (confPercent < 80) confColor = "#F59E0B"; 
  if (confPercent < 50) confColor = "#EF4444"; 

  return (
    <div className="evidence-card fade-in">
      <div className="evidence-header">
        <AlertCircle size={16} />
        <span>Evidence & Provenance</span>
      </div>
      <div className="evidence-body">
        <div className="evidence-row">
          <strong>Sources:</strong> 
          {sources && sources.length > 0 ? (
            sources.map((s, i) => <span key={i} className="badge">{s}</span>)
          ) : (
            <span className="badge error">Unknown</span>
          )}
        </div>
        <div className="evidence-row">
          <strong>Confidence:</strong>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${confPercent}%`, backgroundColor: confColor }}></div>
          </div>
          <span className="conf-text">{confPercent}%</span>
        </div>
        <div className="evidence-row">
          <strong>Provenance:</strong> <span className="provenance-text">{provenance}</span>
        </div>
      </div>
    </div>
  );
}

const formatYAxis = (tickItem) => {
  if (Math.abs(tickItem) >= 1000000) {
    return (tickItem / 1000000).toFixed(1) + 'M';
  }
  if (Math.abs(tickItem) >= 1000) {
    return (tickItem / 1000).toFixed(1) + 'k';
  }
  return tickItem;
};

function ChartRenderer({ chartData }) {
  if (!chartData || !chartData.render_chart || !chartData.data || chartData.data.length === 0) return null;

  return (
    <div className="chart-container fade-in">
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData.data} margin={{ top: 20, right: 30, left: 20, bottom: 70 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            angle={-45} 
            textAnchor="end"
            interval={0}
            tick={{ fontSize: 12 }}
            height={70}
          />
          <YAxis tickFormatter={formatYAxis} />
          <Tooltip 
            formatter={(value) => new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(value)}
          />
          <Legend verticalAlign="top" height={36}/>
          <Bar dataKey="value" fill="#4F46E5" name="Value" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function SuggestedQuestions({ questions, onSelect }) {
  if (!questions || questions.length === 0) return null;
  
  return (
    <div className="suggested-questions fade-in">
      <p className="suggested-title">Follow-up questions:</p>
      <div className="suggested-list">
        {questions.map((q, i) => (
          <button key={i} className="suggested-pill" onClick={() => onSelect(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [typingIndex, setTypingIndex] = useState(-1);
  const [currentSessionId, setCurrentSessionId] = useState(() => localStorage.getItem('currentSessionId') || null);
  const [sessions, setSessions] = useState([]);
  const messagesEndRef = useRef(null);

  const fetchSessions = async () => {
    try {
      const response = await axios.get(`${API_URL}s`); // GET /chats
      setSessions(response.data);
    } catch (error) {
      console.error("Error fetching sessions", error);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem('currentSessionId', currentSessionId);
    }
  }, [currentSessionId]);

  const loadSession = async (sessionId) => {
    if (loading) return;
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}s/${sessionId}`); // GET /chats/{sessionId}
      const loadedMessages = response.data.messages.map(m => {
        let textAnswer = "";
        let evidenceObj = undefined;
        let chartObj = undefined;
        let suggestObj = undefined;

        if (m.role === 'assistant') {
          if (typeof m.content === 'string') {
            textAnswer = m.content;
          } else if (typeof m.content === 'object' && m.content !== null) {
            textAnswer = m.content.text_answer || "";
            evidenceObj = m.content.evidence;
            chartObj = m.content.chart_data;
            suggestObj = m.content.suggested_questions;
          }
        }

        return {
          role: m.role,
          content: m.role === 'user' ? m.content : undefined,
          text_answer: m.role === 'assistant' ? textAnswer : undefined,
          evidence: evidenceObj,
          chart_data: chartObj,
          suggested_questions: suggestObj
        };
      });
      setMessages(loadedMessages);
      setCurrentSessionId(sessionId);
    } catch (error) {
      console.error("Error loading session", error);
    } finally {
      setLoading(false);
    }
  };

  const startNewSession = () => {
    setCurrentSessionId(null);
    localStorage.removeItem('currentSessionId');
    setMessages([]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, typingIndex]);

  const handleSend = async (textOverride) => {
    const textToSend = typeof textOverride === 'string' ? textOverride : input;
    if (!textToSend.trim()) return;

    const userMessage = { role: 'user', content: textToSend };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    scrollToBottom();

    try {
      const apiMessages = newMessages.map(m => ({ role: m.role, content: m.content || m.text_answer || "" }));
      
      const payload = { messages: apiMessages };
      if (currentSessionId) {
        payload.session_id = currentSessionId;
      }
      
      const response = await axios.post(API_URL, payload);
      const data = response.data;

      if (data.session_id && data.session_id !== currentSessionId) {
        setCurrentSessionId(data.session_id);
        fetchSessions();
      }

      setMessages([...newMessages, { 
        role: 'assistant', 
        text_answer: data.text_answer,
        evidence: data.evidence,
        chart_data: data.chart_data,
        suggested_questions: data.suggested_questions
      }]);
      setTypingIndex(-1); // Instantly complete since typewriter is removed
    } catch (error) {
      console.error("Error communicating with agent backend:", error);
      setMessages([...newMessages, { 
        role: 'assistant', 
        text_answer: "Sorry, I encountered an error communicating with the backend. Please check the terminal for errors." 
      }]);
      setTypingIndex(-1);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <span>Chat History</span>
          <button className="new-chat-btn" onClick={startNewSession}>New Chat</button>
        </div>
        {sessions.map(s => (
          <div 
            key={s.session_id} 
            className={`session-item ${currentSessionId === s.session_id ? 'active' : ''}`}
            onClick={() => loadSession(s.session_id)}
          >
            {s.title || "New Chat"}
          </div>
        ))}
      </div>
      
      <div className="main-content">
        <header className="app-header">
          <h1>Intelligent Business Agent</h1>
          <p>Ask me about Churn Risk & Forecasts</p>
          <div className="limitation-notice">Note: Depending on prompt complexity, loading time may be high.</div>
        </header>
        
        <main className="chat-container">
          <div className="messages-list">
            {messages.length === 0 && (
              <div className="empty-state">
                <p>Hello! I am your AI Business Assistant</p>
              </div>
            )}
            {messages.map((msg, index) => {
              const isTyping = typingIndex === index;
              
              return (
                <div key={index} className={`message-wrapper ${msg.role}`}>
                  <div className="message-avatar">
                    {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className="message-content">
                    <div className="message-text markdown-body">
                      {msg.role === 'user' ? (
                        msg.content
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{msg.text_answer}</ReactMarkdown>
                      )}
                    </div>
                    
                    {msg.role === 'assistant' && (
                      <>
                        <ChartRenderer chartData={msg.chart_data} />
                        <EvidenceCard evidence={msg.evidence} />
                        <SuggestedQuestions 
                          questions={msg.suggested_questions} 
                          onSelect={(q) => handleSend(q)} 
                        />
                      </>
                    )}
                  </div>
                </div>
              );
            })}
            {loading && (
              <div className="message-wrapper assistant">
                <div className="message-avatar"><Bot size={20} /></div>
                <div className="message-content">
                  <div className="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className="input-area">
            <input 
              type="text" 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Type your question here..."
              disabled={loading}
            />
            <button onClick={handleSend} disabled={loading || !input.trim()}>
              <Send size={20} />
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
