// src/components/ResultsPanel.js
import React from 'react';
import ResultCard from './ResultCard';
import Visualisations from './Visualisations'; // Import the new Visualisations component

function ResultsPanel({ results }) {
  if (!results) {
    return (
      <aside className="right-panel card results-panel">
        <h3>RESULTS</h3>
        <div className="results-empty">PRESS ANALYSE</div>
      </aside>
    );
  }

  return (
    <aside className="right-panel card results-panel">
      <h3>RESULTS</h3>
      <div>
        <div className="muted">{results.summary}</div>

        {/* Render the visualisations component */}
        <Visualisations results={results} />

        <h4 style={{marginTop: '24px', marginBottom: '0px'}}>Details</h4>
        {results.items.map((item) => (
          <ResultCard key={item.id} item={item} />
        ))}
      </div>
    </aside>
  );
}

export default ResultsPanel;