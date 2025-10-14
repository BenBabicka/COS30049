// src/App.js
import React, { useState, useEffect } from 'react';
import './styles.css';
import InputPanel from './components/InputPanel';
import ResultsPanel from './components/ResultsPanel';

function App() {
  const [tab, setTab] = useState('link');
  const [items, setItems] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // State for managing dark mode
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage for a saved theme preference
    const savedMode = localStorage.getItem('isDarkMode');
    return savedMode ? JSON.parse(savedMode) : false;
  });

  // Effect to update localStorage and the body class when isDarkMode changes
  useEffect(() => {
    localStorage.setItem('isDarkMode', JSON.stringify(isDarkMode));
    if (isDarkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [isDarkMode]);

  const handleAnalyse = () => {
    if (items.length === 0) {
      alert('Add at least one item to analyse');
      return;
    }
    setLoading(true);

    setTimeout(() => {
      const classifications = ['Misinformation', 'Legitimate'];
      const demoResults = {
        summary: `Analysis complete for ${items.length} item(s).`,
        items: items.map((it, idx) => ({
          id: idx + 1,
          type: it.type,
          value: it.value,
          classification: classifications[Math.floor(Math.random() * classifications.length)],
          confidence: Math.random() * 0.4 + 0.6,
          result: 'Analysis Complete',
        })),
      };
      setResults(demoResults);
      setItems([]);
      setLoading(false);
    }, 500);
  };

  const handleReset = () => {
    setItems([]);
    setResults(null);
    setTab('link');
    setLoading(false);
  };
  
  const toggleDarkMode = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  return (
    <>
      <header>
        <div>
          <h1>COVID-19 Misinformation Detector</h1>
          <div className="muted">Detect COVID-19 misinformation in links, text, or uploads.</div>
        </div>
        <button onClick={toggleDarkMode} className="theme-toggle-btn">
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
      </header>

      <main className="layout">
        <div className="left-panel card">
          <div className="panel-header">
            <h2>COVID-19 MISINFORMATION DETECTOR</h2>
            <button className="reset-btn" onClick={handleReset}>RESET</button>
          </div>
          <InputPanel
            tab={tab}
            setTab={setTab}
            items={items}
            setItems={setItems}
            onAnalyse={handleAnalyse}
            loading={loading}
          />
        </div>
        <ResultsPanel results={results} />
      </main>
    </>
  );
}

export default App;