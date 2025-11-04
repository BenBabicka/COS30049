// src/App.js
import React, { useState, useEffect } from 'react';
import './styles.css';
import InputPanel from './components/InputPanel';
import ResultsPanel from './components/ResultsPanel';
import ModelStats from "./components/ModelStats";

const BACKEND_URL = 'http://localhost:8000'; // FastAPI backend URL

function App() {
  const [tab, setTab] = useState('text'); // Current input tab ('link', 'text', 'upload')
  const [items, setItems] = useState([]); // Stores items to be analyzed: { type: 'text'/'link', value: '...' }
  const [results, setResults] = useState(null); // Stores analysis results from the backend
  const [loading, setLoading] = useState(false); // Indicates if analysis is in progress
  const [isDarkMode, setIsDarkMode] = useState(false); // State for theme toggling

  // Effect to apply the dark mode class to the body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Function to handle the analysis request to the backend
  const handleAnalyse = async () => {
    if (items.length === 0) {
      alert('Add at least one item to analyse');
      return;
    }
    setLoading(true);
    setResults(null); // Clear previous results

    // Prepare data in the format expected by the backend: { "data": [...] }
    const dataToSend = {
      data: items // Extract only the 'value' strings
    };

    try {
      // Send the formatted data to the backend's /use endpoint
      const response = await fetch(`${BACKEND_URL}/use`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json', // Specify expected response type
        },
        body: JSON.stringify(dataToSend),
      });

      // Handle HTTP errors
      if (!response.ok) {
        let errorDetail = `HTTP error! status: ${response.status}`;
        try {
          // Try to get more specific error details from the backend response
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (jsonError) {
          // Fallback if parsing error JSON fails
          errorDetail = response.statusText || errorDetail;
        }
        throw new Error(errorDetail);
      }

      // Parse the backend's response (expects a list of tuples/arrays: [["real", 0.85], ...])
      const backendResponseData = await response.json();

      // Validate the structure of the backend response
      if (!backendResponseData.type === 'dict') {
         throw new Error("Received invalid data format from backend (expected an array).");
      }

      const finalResults = []

      for (const [key, value] of Object.entries(backendResponseData)) {
      const id = key;
        const tweet = value['text'];
        const type = value['type'];
      const classification = value['classification-response'];
      const confidence = value['regression-response'];
      finalResults.push({"id":id,'type':type.toUpperCase(), "tweet":tweet, "classification":classification, "confidence":confidence });
      }
      setResults(finalResults); // Update state with the analysis results
      setItems([]); // Clear input items list on successful analysis

    } catch (error) {
      console.error("Error calling backend or processing response:", error);
      alert(`Analysis failed: ${error.message}`); // Show error message to the user
      // Keep items in the input list on failure so the user can retry or edit
    } finally {
      setLoading(false); // Stop the loading indicator
    }
  };

  // Function to reset the application state
  const handleReset = () => {
    setItems([]);       // Clear input items
    setResults(null);     // Clear results
    setTab('text');       // Reset to the default tab
    setLoading(false);    // Ensure loading indicator is off
   };

  // Function to toggle the dark mode state
  const toggleDarkMode = () => {
    setIsDarkMode(prevMode => !prevMode);
  };

  return (
    <>
      <header>
        <h1>Misinformation Detector</h1>
        {/* Toggle Button */}
        <button onClick={toggleDarkMode} className="theme-toggle-btn">
          Toggle {isDarkMode ? 'Light' : 'Dark'} Mode
        </button>
      </header>
      <main className="layout">
          <div className="left-side">
        <div className="left-panel card">
          <div className="panel-header">
             <h2>INPUT</h2>
             <button onClick={handleReset} className="reset-btn">Reset</button>
          </div>
          {/* InputPanel handles adding/removing items and triggering analysis */}
          <InputPanel
            tab={tab}
            setTab={setTab}
            items={items}
            setItems={setItems}
            onAnalyse={handleAnalyse}
            loading={loading}
          />
        </div>
          <ModelStats></ModelStats>
          </div>
          {/* ResultsPanel displays the analysis results */}
        <ResultsPanel results={results} />
      </main>
    </>
  );
}
export default App;