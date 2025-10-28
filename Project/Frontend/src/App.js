// src/App.js
import React, { useState, useEffect } from 'react';
import './styles.css';
import InputPanel from './components/InputPanel';
import ResultsPanel from './components/ResultsPanel';

const BACKEND_URL = 'http://localhost:8000'; // FastAPI backend URL

function App() {
  const [tab, setTab] = useState('link'); // Current input tab ('link', 'text', 'upload')
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
      data: items.map(item => item.value) // Extract only the 'value' strings
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
      if (!Array.isArray(backendResponseData)) {
         throw new Error("Received invalid data format from backend (expected an array).");
      }
      if (backendResponseData.length !== items.length) {
         // Warn if the number of results doesn't match the number of items sent
         console.warn("Mismatch between sent items and received results count.");
         // Note: The code proceeds, assuming results correspond to the first N items.
         // You might want stricter handling here depending on requirements.
      }

      // Transform the backend response array into the structure needed by the frontend ResultsPanel
      const transformedItems = backendResponseData.map((resultTuple, index) => {
          // Handle potential malformed tuples in the response
          if (!Array.isArray(resultTuple) || resultTuple.length < 2) {
              console.error(`Invalid result tuple at index ${index}:`, resultTuple);
              // Provide a default error structure for this item
              return {
                  id: index + 1,
                  type: items[index]?.type || 'unknown', // Use original item type if available
                  value: items[index]?.value || 'Error processing item', // Use original value if available
                  classification: 'error', // Indicate an error state
                  confidence: 0,
                  result: 'Invalid response from backend'
              };
          }

          // Extract classification and confidence from the tuple
          const [classification, confidence] = resultTuple;
          // Get the original item corresponding to this result
          const originalItem = items[index] || {}; // Fallback for safety

          // Create the result item structure for the frontend
          return {
              id: index + 1, // Simple sequential ID
              type: originalItem.type || 'text', // Use original type, default to 'text'
              value: originalItem.value || 'N/A', // Use original value
              classification: classification, // Keep backend format ("real"/"fake")
              confidence: confidence, // Use confidence score from backend
              result: 'Analysis Complete' // Status message for the item
          };
      });

      // Structure the final results object for the ResultsPanel
      const finalResults = {
          summary: `Analysis complete for ${transformedItems.length} item(s).`,
          items: transformedItems
      };

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
    setTab('link');       // Reset to the default tab
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
        {/* ResultsPanel displays the analysis results */}
        <ResultsPanel results={results} />
      </main>
    </>
  );
}
export default App;