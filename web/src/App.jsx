import React, { useState } from "react";

export default function App() {
  const [idx, setIdx] = useState(-1);
  const [display, setDisplay] = useState(null);   
  const [features, setFeatures] = useState(null); 
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const loadClient = async (next = false) => {
    setError("");
    setResult(null);
    const i = next ? idx + 1 : 0;
    const res = await fetch(`/reference/client?i=${i}`);
    if (!res.ok) {
      const e = await res.text();
      setError(e || res.statusText);
      return;
    }
    const data = await res.json();
    if (!data.display) {
      setError("Plus de clients dans la référence.");
      return;
    }
    setIdx(data.index);
    setDisplay(data.display);
    setFeatures(data.features);
  };

  const predict = async () => {
    setError("");
    setResult(null);
    if (!features) {
      setError("Aucun client chargé.");
      return;
    }
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([features]),
    });
    if (!res.ok) {
      const e = await res.text();
      setError(e || res.statusText);
      return;
    }
    const data = await res.json();
    const p = data.y_proba?.[0] ?? 0;
    const y = data.y_pred?.[0] ?? 0;
    setResult({
      y_pred: y,
      y_proba: p,
      label: y === 1 ? "Potentiel bon payeur" : "Potentiel mauvais payeur",
      threshold: data.threshold,
    });
  };

  const nextClient = async () => {
    await loadClient(true);
  };

  const InfoLine = ({ label, value }) => (
    <div style={{ display: "flex", gap: 8, marginBottom: 6 }}>
      <div style={{ width: 250, fontWeight: 600 }}>{label}</div>
      <div>{value ?? "-"}</div>
    </div>
  );

  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "Segoe UI, sans-serif" }}>
      <h1>Score Octroi - Démo Opérationnelle</h1>

      <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
        <button onClick={() => loadClient(false)}>Nouveau client</button>
        <button onClick={nextClient} disabled={idx < 0}>Client suivant</button>
        <button onClick={predict} disabled={!features}>Prédire le risque</button>
      </div>

      {display && (
        <div style={{ border: "1px solid #ddd", borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <h3 style={{ marginTop: 0 }}>Client #{idx + 1}</h3>
          <InfoLine label="Âge" value={display.Age_Tit} />
          <InfoLine label="Ressources" value={display.Ressource} />
          <InfoLine label="Ancienneté bancaire" value={display.Ancien_Banc_Tit} />
          <InfoLine label="Revenu" value={display.Mrev_Tit} />
          <InfoLine label="Ancienneté professionnelle" value={display.Ancien_Prof_Tit} />
          <InfoLine label="Montant des impôts" value={display.ZCOM_SR_CL_MIMPOTS} />
          <InfoLine label="Charges" value={display.Charge} />
          <InfoLine label="Taux reste à vivre" value={display.Ratio_Ress_RAV} />
          <InfoLine label="Habitation" value={display.MCLFCHAB1} />
          <InfoLine label="Situation familiale" value={display.MCLFCSITFAM} />
          <InfoLine label="CSP" value={display.CSP_Tit} />
        </div>
      )}

      {result && (
        <div style={{ border: "1px solid #cbd5e1", background: "#f8fafc", borderRadius: 8, padding: 16, marginBottom: 16 }}>
          <h3 style={{ marginTop: 0 }}>Résultat</h3>
          <InfoLine label="Prédiction" value={`${result.y_pred} : ${result.label}`} />
          <InfoLine label="Probabilité" value={result.y_proba?.toFixed(4)} />
          <InfoLine label="Seuil" value={result.threshold} />
        </div>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        <button onClick={nextClient} disabled={idx < 0}>Accepter client</button>
        <button onClick={nextClient} disabled={idx < 0}>Refuser client</button>
      </div>

      {error && <pre style={{ color: "crimson", marginTop: 16 }}>{String(error)}</pre>}
    </div>
  );
}