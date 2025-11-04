// src/components/ResultsPanel.js
import React from 'react';
import ResultCard from './ResultCard'; // Component to display individual result items
import Visualisations from './Visualisations'; // Component for charts

// ResultsPanel component: Displays the analysis results, including summary, visualizations, and individual item details.
function ResultsPanel({ results }) {
  // If no results are available yet, show a placeholder message.
  if (!results) {
    return (
      <aside className="right-panel card results-panel">
        <h3>RESULTS</h3>
        <div className="results-empty">PRESS ANALYSE</div>
      </aside>
    );
  }

  // If results are available, display the summary, visualizations, and result cards.
  return (
    <aside className="right-panel card results-panel">
      <h3>RESULTS</h3>
      <div>
        {/* Display the summary message */}
        <div className="muted">{results.summary}</div>

        {/* Render the visualisations component (Doughnut and Bar charts) */}
        {/* The Visualisations component uses the 'results' object to generate charts */}
        <Visualisations results={results} />

        {/* Section header for detailed results */}
        <h4 style={{marginTop: '24px', marginBottom: '0px'}}>Details</h4>

        {/* Map through each item in the results and render a ResultCard for it */}
        {/* The ResultCard component handles the display logic for each individual item type and classification */}
        {results.map((item) => (
          <ResultCard key={item["id"]} item={item} /> // Use item.id as the key
        ))}
      </div>
    </aside>
  );
}

export default ResultsPanel;