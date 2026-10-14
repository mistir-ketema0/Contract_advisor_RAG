import { useEffect, useRef, useState } from "react";
import { api } from "./api";

const suggestions = [
  "What is the termination notice period?",
  "When must invoices be paid?",
  "Which law governs this agreement?",
];

function Answer({ result }) {
  return (
    <article className="answer-card">
      <div className="answer-heading">
        <span className="spark">✳</span>
        <strong>Contract Advisor</strong>
        <span
          className={`badge ${result.insufficient_evidence ? "muted" : ""}`}
        >
          {result.insufficient_evidence
            ? "Insufficient evidence"
            : "Source attached"}
        </span>
      </div>
      <p className="answer-text">{result.answer}</p>
      {result.citations.length > 0 && (
        <div className="sources">
          <h3>
            Check the evidence{" "}
            <span>
              {result.citations.length} source
              {result.citations.length > 1 ? "s" : ""}
            </span>
          </h3>
          {result.citations.map((citation) => (
            <details key={citation.number} open={result.citations.length === 1}>
              <summary>
                <span className="source-number">{citation.number}</span>
                <span>
                  <strong>{citation.section}</strong>
                  <small>
                    {citation.source} · Page {citation.page}
                  </small>
                </span>
                <span className="expand">↗</span>
              </summary>
              <blockquote>{citation.quote}</blockquote>
            </details>
          ))}
        </div>
      )}
      <div className="answer-footer">
        {result.mode === "demo"
          ? "Extractive demo · No AI model used"
          : "AI-generated · Verify against the source"}
        <span>{(result.timings_ms.total / 1000).toFixed(2)}s</span>
      </div>
    </article>
  );
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [mode, setMode] = useState(null);
  const [embeddingTag, setEmbeddingTag] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileInput = useRef(null);
  const end = useRef(null);

  useEffect(() => {
    let active = true;
    Promise.all([api("/health"), api("/documents")])
      .then(([health, docs]) => {
        if (!active) return;
        setMode(health.mode);
        setEmbeddingTag(health.embedding_tag);
        setDocuments(docs);
        const compatible = docs.filter(
          (d) => d.embedding_tag === health.embedding_tag,
        );
        setSelected(compatible.length ? [compatible[0].id] : []);
      })
      .catch((e) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, busy]);

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const result = await api("/documents", { method: "POST", body });
      setDocuments(await api("/documents"));
      setSelected([result.document.id]);
      setMessages([]);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function remove(document) {
    if (
      !window.confirm(
        `Remove ${document.filename} and its stored text from this workspace?`,
      )
    )
      return;
    setUploading(true);
    setError("");
    try {
      await api(`/documents/${document.id}`, { method: "DELETE" });
      setDocuments((prev) => prev.filter((d) => d.id !== document.id));
      setSelected((prev) => prev.filter((id) => id !== document.id));
      setMessages([]);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function ask(event) {
    event.preventDefault();
    if (!question.trim() || !selected.length || busy || uploading) return;
    const text = question.trim();
    setQuestion("");
    setError("");
    setBusy(true);
    setMessages((prev) => [...prev, { kind: "question", text }]);
    try {
      const result = await api("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, document_ids: selected }),
      });
      setMessages((prev) => [...prev, { kind: "answer", result }]);
    } catch (e) {
      setError(e.message);
      setQuestion(text);
    } finally {
      setBusy(false);
    }
  }

  function toggle(id) {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id],
    );
    setMessages([]);
  }

  return (
    <div className="workspace">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Contract Advisor home">
          <span className="brand-mark">C</span>
          <span>
            Contract<span className="brand-light">Advisor</span>
          </span>
        </a>
        <div className="library-heading">
          <h2>Your contracts</h2>
          <span>{documents.length}</span>
        </div>
        <p className="sidebar-intro">
          Choose the documents your answers should come from.
        </p>
        <input
          ref={fileInput}
          type="file"
          accept=".pdf,.txt,.md"
          onChange={upload}
          className="file-input"
          aria-label="Upload contract"
        />
        <button
          className="upload-button"
          onClick={() => fileInput.current.click()}
          disabled={uploading || busy}
        >
          <span>＋</span>
          {uploading ? "Indexing document…" : "Add a contract"}
        </button>
        <p className="file-hint">PDF, TXT or MD · Up to 10 MB</p>
        <div className="document-list">
          {documents.length === 0 && (
            <div className="empty-library">
              No contracts yet.
              <br />
              Add the included sample to try it.
            </div>
          )}
          {documents.map((doc) => {
            const incompatible = doc.embedding_tag !== embeddingTag;
            return (
              <div
                className={`document ${selected.includes(doc.id) ? "selected" : ""}`}
                key={doc.id}
              >
                <label>
                  <input
                    type="checkbox"
                    checked={selected.includes(doc.id)}
                    disabled={busy || uploading || incompatible}
                    onChange={() => toggle(doc.id)}
                  />
                  <span className="document-icon">▤</span>
                  <span className="document-name">
                    <strong>{doc.filename}</strong>
                    <small>
                      {incompatible
                        ? "Re-upload for this mode"
                        : `${doc.page_count} pages · Indexed`}
                    </small>
                  </span>
                </label>
                <button
                  className="remove"
                  aria-label={`Remove ${doc.filename}`}
                  onClick={() => remove(doc)}
                  disabled={busy || uploading}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
        <div className="sidebar-bottom">
          <span className="status-dot"></span>
          {mode
            ? mode === "demo"
              ? "Offline demo"
              : "AI connected"
            : "Connecting…"}
          <p>
            Source text stays in this workspace in demo mode. AI mode sends text
            to the model provider.
          </p>
        </div>
      </aside>
      <main>
        <header>
          <span className="eyebrow">THE CONTRACT WORKSPACE</span>
          <div className="header-actions">
            <span className="badge muted">{selected.length} selected</span>
            <button
              className="reset"
              disabled={busy || !messages.length}
              onClick={() => setMessages([])}
            >
              Clear conversation
            </button>
          </div>
        </header>
        <div className="content">
          {messages.length === 0 ? (
            <section className="welcome">
              <div className="welcome-icon">↗</div>
              <p className="eyebrow">READ LESS. UNDERSTAND MORE.</p>
              <h1>
                Your contracts,
                <br />
                <em>with the evidence.</em>
              </h1>
              <p className="welcome-copy">
                Ask a question. Inspect the clause behind the answer.
                <br />
                Keep every conclusion connected to its source.
              </p>
              <div className="suggestions">
                {suggestions.map((text) => (
                  <button
                    key={text}
                    disabled={!selected.length || uploading}
                    onClick={() => setQuestion(text)}
                  >
                    {text}
                    <span>↗</span>
                  </button>
                ))}
              </div>
              {mode === "demo" && (
                <p className="demo-note">
                  Demo mode shows matching source passages. Enable AI mode for
                  generated answers.
                </p>
              )}
            </section>
          ) : (
            <section className="conversation" aria-live="polite">
              {messages.map((message, i) =>
                message.kind === "question" ? (
                  <div className="question-card" key={i}>
                    <span className="eyebrow">YOUR QUESTION</span>
                    <p>{message.text}</p>
                  </div>
                ) : (
                  <Answer key={i} result={message.result} />
                ),
              )}
              {busy && (
                <div className="thinking" role="status">
                  Searching the selected contracts…
                </div>
              )}
              <div ref={end} />
            </section>
          )}
        </div>
        <div className="composer-area">
          {error && (
            <div className="error" role="alert">
              {error}
            </div>
          )}
          <form className="composer" onSubmit={ask}>
            <label className="sr-only" htmlFor="question">
              Ask about your contracts
            </label>
            <input
              id="question"
              value={question}
              maxLength={2000}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={
                selected.length
                  ? "Ask about your selected contracts…"
                  : "Add and select a contract to begin…"
              }
              disabled={busy || uploading}
            />
            <button
              type="submit"
              aria-label="Send question"
              disabled={
                !question.trim() || !selected.length || busy || uploading
              }
            >
              ↑
            </button>
          </form>
          <p className="composer-note">
            Each question is independent. Answers can be incomplete or wrong.
            Verify the source; this is not legal advice.
          </p>
        </div>
      </main>
    </div>
  );
}
