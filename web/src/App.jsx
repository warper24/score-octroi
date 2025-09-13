import React, { useState } from "react";

export default function App() {
  const [json, setJson] = useState(`[{"Mrev_Tit":0}]`);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const predict = async () => {
    setError("");
    setResult(null);
    try {
      const payload = JSON.parse(json);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const e = await res.json().catch(()=>({detail:res.statusText}));
        throw new Error(e.detail || "HTTP " + res.status);
      }
      setResult(await res.json());
    } catch (e) {
      setError(String(e.message || e));
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>Score Octroi - Démo</h1>
      <p>Collez un tableau JSON de dossiers, puis lancez la prédiction.</p>
      <textarea
        style={{ width: "100%", height: 200 }}
        value={json}
        onChange={(e) => setJson(e.target.value)}
      />
      <div style={{ marginTop: 12 }}>
        <button onClick={predict}>Predict</button>
      </div>
      {error && <pre style={{ color: "crimson" }}>{error}</pre>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}