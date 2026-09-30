import { useEffect, useState } from "react";

const SOURCE_LABEL = {
  SEMANTIC: "Semantic Search",
  STRUCTURAL: "Structural Analysis",
  CALL_GRAPH: "Call Graph",
  AST: "AST",
  OTHER: "Other"
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const b = await res.json();
      msg = Array.isArray(b.detail) ? b.detail.map((d) => d.msg).join("; ") : b.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function Evidence({ e }) {
  const lines = e.start_line == null ? "n/a" : e.end_line && e.end_line !== e.start_line ? `${e.start_line} - ${e.end_line}` : `${e.start_line}`;
  return (
    <div className="evidence">
      <dl>
        <dt>File</dt><dd className="mono">{e.file_path}</dd>
        <dt>Lines</dt><dd className="mono">{lines}</dd>
        <dt>Source</dt><dd>{SOURCE_LABEL[e.source_type] || e.source_type}{e.symbol ? ` • ${e.symbol}` : ""}</dd>
      </dl>
      {e.is_mock && <div className="mockbadge">MOCK / DEVELOPMENT DATA — not real repository evidence</div>}
      <pre><code>{e.snippet}</code></pre>
    </div>
  );
}

export default function App() {
  const [repo, setRepo] = useState(null);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/api/repository").then(setRepo).catch(() => setRepo({ mode: "UNREACHABLE" }));
  }, []);

  async function run() {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await api("/api/investigate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const inv = result?.investigation;
  return (
    <div className="app">
      <header>
        <h1>Agentic Code Intelligence</h1>
        <span className={`mode ${repo?.mode}`}>
          {repo ? (repo.mode === "UNREACHABLE" ? "Backend unreachable" : repo.mode === "MOCK" ? "Mock mode" : `Repository: ${repo.repo_path}`) : "Connecting..."}
        </span>
      </header>

      <section>
        <textarea
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Where is the Bluetooth settings deep link used?"
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && question.trim() && !loading) run();
          }}
        />
        <button onClick={run} disabled={loading || !question.trim()}>
          {loading ? "Investigating..." : "Investigate"}
        </button>
      </section>

      {error && <div className="error" role="alert">{error}</div>}

      {result && (
        <>
          {result.mode === "MOCK" && (
            <div className="mockbanner">Development mode: all evidence below is mock data.</div>
          )}
          {result.errors?.length > 0 && (
            <div className="warn">
              <strong>Issues during investigation</strong>
              <ul>{result.errors.map((m, i) => <li key={i}>{m}</li>)}</ul>
            </div>
          )}

          <section>
            <h2>Answer</h2>
            <p className="answer">{result.answer}</p>
          </section>

          <section>
            <h2>Evidence ({result.evidence.length})</h2>
            {result.evidence.length === 0 ? (
              <p className="muted">No evidence found.</p>
            ) : (
              result.evidence.map((e, i) => <Evidence key={i} e={e} />)
            )}
          </section>

          <section>
            <h2>Investigation</h2>
            <dl className="grid">
              <dt>Query type</dt><dd>{inv.query_type}</dd>
              <dt>Verification</dt><dd className={inv.verification_status}>{inv.verification_status} (confidence {inv.confidence})</dd>
              <dt>Sources used</dt><dd>{inv.sources_used.join(", ") || "none"}</dd>
            </dl>
            <ol className="steps">
              {inv.steps.map((s) => (
                <li key={s.id}>
                  <span className="mono">{s.capability}</span> — {s.description} <em>[{s.status}]</em>
                </li>
              ))}
            </ol>
            {inv.missing_evidence.length > 0 && (
              <>
                <h3>Missing evidence</h3>
                <ul>{inv.missing_evidence.map((m, i) => <li key={i}>{m}</li>)}</ul>
              </>
            )}
            <details>
              <summary>Execution log</summary>
              <pre>{inv.log.join("\n")}</pre>
            </details>
          </section>
        </>
      )}
    </div>
  );
}