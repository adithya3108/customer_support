import { useEffect, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const MODE_PRESETS = {
  strict: { temperature: 0.2, max_tokens: 150 },
  friendly: { temperature: 0.7, max_tokens: 200 },
};

const EXAMPLES = [
  'My product arrived damaged. Can I get a replacement?',
  'The order was late by a week. What can I do?',
  'I was charged but no order was created.',
];

export default function App() {
  const [complaint, setComplaint] = useState(EXAMPLES[0]);
  const [mode, setMode] = useState('strict');
  const [temperature, setTemperature] = useState(MODE_PRESETS.strict.temperature);
  const [maxTokens, setMaxTokens] = useState(MODE_PRESETS.strict.max_tokens);
  const [response, setResponse] = useState('');
  const [documents, setDocuments] = useState([]);
  const [prompt, setPrompt] = useState('');
  const [provider, setProvider] = useState('');
  const [fallbackUsed, setFallbackUsed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  useEffect(() => {
    const preset = MODE_PRESETS[mode];
    setTemperature(preset.temperature);
    setMaxTokens(preset.max_tokens);
  }, [mode]);

  const stats = useMemo(() => {
    return [
      { label: 'Mode', value: mode === 'strict' ? 'Strict policy' : 'Friendly tone' },
      { label: 'Temp', value: temperature.toFixed(1) },
      { label: 'Tokens', value: maxTokens },
    ];
  }, [mode, temperature, maxTokens]);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError('');
    setResponse('');
    setDocuments([]);
    setPrompt('');
    setProvider('');
    setFallbackUsed(false);

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          complaint,
          mode,
          temperature,
          max_tokens: maxTokens,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Request failed');
      }

      setResponse(data.response);
      setDocuments(data.documents || []);
      setPrompt(data.prompt || '');
      setProvider(data.provider || '');
      setFallbackUsed(Boolean(data.fallback_used));
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Day 11</p>
          <h1>AI-Assisted Customer Support Response Generator</h1>
          <p className="lede">
            BM25 retrieves the most relevant policy snippets from a local dataset, then an LLM
            drafts a policy-based customer support reply.
          </p>
        </div>
        <div className="hero-card">
          <span className={`status-dot ${health?.status === 'ok' ? 'ok' : 'bad'}`} />
          <div>
            <strong>Backend</strong>
            <p>{health?.status === 'ok' ? `Online · ${health.provider || 'unknown provider'}` : 'Offline'}</p>
          </div>
        </div>
      </section>

      <main className="grid">
        <section className="panel editor">
          <form onSubmit={handleSubmit} className="form">
            <label>
              Customer complaint
              <textarea
                value={complaint}
                onChange={(e) => setComplaint(e.target.value)}
                placeholder="Describe the customer issue..."
              />
            </label>

            <div className="row">
              <label>
                Mode
                <select value={mode} onChange={(e) => setMode(e.target.value)}>
                  <option value="strict">Strict policy</option>
                  <option value="friendly">Friendly tone</option>
                </select>
              </label>

              <label>
                Temperature
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                />
              </label>

              <label>
                Max tokens
                <input
                  type="number"
                  min="50"
                  max="1000"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                />
              </label>
            </div>

            <div className="example-row">
              {EXAMPLES.map((sample) => (
                <button
                  key={sample}
                  type="button"
                  className="ghost"
                  onClick={() => setComplaint(sample)}
                >
                  Use example
                </button>
              ))}
            </div>

            <button className="primary" type="submit" disabled={loading}>
              {loading ? 'Generating...' : 'Generate response'}
            </button>
          </form>
        </section>

        <aside className="panel summary">
          <div className="stat-row">
            {stats.map((item) => (
              <div key={item.label} className="stat-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className="helper">
            <p className="mini-title">Retrieval flow</p>
            <p>Local policies → BM25 top 3 → prompt template → LLM response</p>
            <p>Fallback mode returns an escalation message when no policy match is strong enough.</p>
          </div>
        </aside>
      </main>

      <section className="panel results">
        <div className="section-head">
          <div>
            <p className="eyebrow">Output</p>
            <h2>Draft response and sources</h2>
          </div>
          <div className="pill-group">
            <span className={`pill ${fallbackUsed ? 'warn' : 'ok'}`}>
              {fallbackUsed ? 'Fallback used' : 'Policy match found'}
            </span>
            {provider ? <span className="pill">{provider}</span> : null}
          </div>
        </div>

        {error ? <div className="error-box">{error}</div> : null}

        <div className="response-box">
          <div className="response-head">AI response</div>
          <p>{response || 'Your response will appear here.'}</p>
        </div>

        <div className="response-box">
          <div className="response-head">Prompt used</div>
          <pre>{prompt || 'Prompt preview will appear here.'}</pre>
        </div>

        <div className="sources">
          <div className="response-head">Retrieved documents</div>
          <div className="source-grid">
            {documents.length ? (
              documents.map((doc) => (
                <article key={doc.title} className="source-card">
                  <div className="source-top">
                    <strong>{doc.title}</strong>
                    <span>Score {doc.score.toFixed(2)}</span>
                  </div>
                  <p>{doc.content}</p>
                </article>
              ))
            ) : (
              <div className="empty-state">Retrieved policies will show here after you generate a response.</div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
