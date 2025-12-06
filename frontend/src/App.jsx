// src/App.jsx
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000/query";

function App() {
  const [mode, setMode] = useState("rag"); // "rag", "kg", or "kg_agent"
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [hasAnswer, setHasAnswer] = useState(false);

  const handleAsk = async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setStatus("Please enter a question.");
      return;
    }

    setLoading(true);
    setStatus("Thinking...");
    setHasAnswer(false);
    setAnswer("");
    setSources([]);

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, question: trimmed }),
      });

      if (!resp.ok) {
        throw new Error("Server error: " + resp.status);
      }

      const data = await resp.json();
      setAnswer(data.answer || "(No answer returned)");
      setSources(Array.isArray(data.sources) ? data.sources : []);
      setHasAnswer(true);
      setStatus("");
    } catch (err) {
      console.error(err);
      setStatus("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const modeLabel =
    mode === "rag"
      ? "Baseline RAG"
      : mode === "kg"
      ? "CT-Engine Cypher"
      : "KG-first Agent";

  return (
    <div className="page">
      <div className="container">
      <h1>TReK Demo</h1>
      <p className="subtitle">
        Choose <strong>Baseline RAG</strong> or <strong>Knowledge Graph</strong>{" "}
        mode and ask a question about Talazoparib resistance.
      </p>

      <div className="mode-toggle">
        <div
          className={`pill ${mode === "rag" ? "active" : ""}`}
          onClick={() => setMode("rag")}
        >
          Baseline RAG
        </div>
        <div
          className={`pill ${mode === "kg" ? "active" : ""}`}
          onClick={() => setMode("kg")}
        >
          KG (Cypher)
        </div>
        <div
          className={`pill ${mode === "kg_agent" ? "active" : ""}`}
          onClick={() => setMode("kg_agent")}
        >
          KG-first Agent
        </div>
      </div>

      <textarea
        placeholder="e.g., Which genes have been implicated in resistance to talazoparib?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <div className="actions">
        <button onClick={handleAsk} disabled={loading}>
          {loading ? "Asking..." : "Ask"}
        </button>
      </div>

      <div className="status">{status}</div>

      {hasAnswer && (
        <div className="answer-card">
          <div className="badge">Mode: {modeLabel}</div>

          <div className="answer-text">
            <ReactMarkdown>{answer}</ReactMarkdown>
          </div>

          {sources && sources.length > 0 && (
            <div className="sources">
              <div className="sources-title">Sources</div>
              <div>
                {sources.map((s, idx) => (
                  <div key={idx} className="source-item">
                    <div className="source-label">
                      {s.label || (s.pmid ? `PMID ${s.pmid}` : "Source")}
                    </div>
                    {s.snippet && (
                      <div className="source-snippet">{s.snippet}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
    </div>
  );
}

export default App;
