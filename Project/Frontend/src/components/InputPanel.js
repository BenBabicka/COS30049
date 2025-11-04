// src/components/InputPanel.js
import React, { useState } from 'react';

// Simple function to escape basic HTML characters to prevent XSS issues when rendering input values
function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str.replace(/[&<>"']/g, function (m) {
    // Map special characters to their HTML entity equivalents
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}

// InputPanel component: Handles user input (links, text, file uploads) and manages the list of items to analyze.
function InputPanel({ tab, setTab, items, setItems, onAnalyse, loading }) {
  const [inputValue, setInputValue] = useState(''); // State for the text input field

  // --- Input Validation ---
  // Validates the input based on the current tab ('link' or 'text')
  const validateInput = (type, value) => {
    // Check for empty or whitespace-only input
    if (!value || value.trim() === '') {
      alert('Input cannot be empty.');
      return false;
    }
    // Specific validation for links using a regex pattern
    if (type === 'link') {
      // Basic URL pattern check
      const urlPattern = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,})([\/\w \.-]*)*\/?$/i;
      if (!urlPattern.test(value)) {
        alert('Please enter a valid URL or domain (e.g., example.com or http://example.com).');
        return false;
      }
    }
    // Input is valid if it passes the checks
    return true;
  };

  // --- Add Item ---
  // Handles adding a new item (link or text) from the input field to the items list
  const handleAddItem = () => {
    // Validate the input before adding
    if (!validateInput(tab, inputValue)) return;
    // Add the new item to the existing items array
    setItems((prevItems) => [...prevItems, { type: tab, value: inputValue.trim() }]);
    // Clear the input field after adding
    setInputValue('');
  };

  // --- Handle Enter Key ---
  // Allows adding items by pressing Enter in the text input field (except for file upload tab)
  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && tab !== 'upload') {
        handleAddItem(); // Call the add item handler
        event.preventDefault(); // Prevent default Enter key behavior (e.g., form submission)
    }
  };


  // --- Remove Item ---
  // Handles removing an item from the list based on its index
  const handleRemoveItem = (indexToRemove) => {
    // Filter out the item at the specified index
    setItems((prevItems) => prevItems.filter((_, index) => index !== indexToRemove));
  };

  // --- File Upload ---
  // Handles file selection, validation, and processing for .txt and .csv files
  const handleFileUpload = (event) => {
    const file = event.target.files && event.target.files[0];
    // Do nothing if no file is selected
    if (!file) return;

    // File type validation
    const allowedTypes = ['text/plain', 'text/csv'];
    if (!allowedTypes.includes(file.type)) {
      alert(`Invalid file type. Please upload a .txt or .csv file.`);
      event.target.value = null; // Reset file input
      return;
    }
    // File size validation (max 5MB)
    const maxSizeInBytes = 5 * 1024 * 1024;
    if (file.size > maxSizeInBytes) {
      alert(`File is too large. Please upload a file smaller than 5MB.`);
      event.target.value = null; // Reset file input
      return;
    }

    // Read File content using FileReader API
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      // Split file content into lines based on newline characters
      const lines = content.split(/\r?\n/);
      // Process each line: trim whitespace, filter out empty lines, and create 'text' items
      const newItems = lines
        .map(line => line.trim())
        .filter(line => line !== '')
        .map(line => ({ type: 'text', value: line })); // Each non-empty line becomes a 'text' item

      // Add the new items from the file to the list
      if (newItems.length > 0) {
        setItems((prevItems) => [...prevItems, ...newItems]);
      } else {
        alert(`The file "${file.name}" is empty or contains only empty lines.`);
      }
    };
    // Handle potential errors during file reading
    reader.onerror = (e) => {
        console.error("Error reading file:", e);
        alert(`Error reading file "${file.name}".`);
    };
    // Read the file as text
    reader.readAsText(file);
    // Reset the file input value to allow uploading the same file again if needed
    event.target.value = null;
  };

  // --- Render ---
  // Renders the InputPanel UI, including tabs, input field/button, items list, and analyse button
  return (
    <>
      {/* Tabs for selecting input type */}
      <div className="tabs">
          {/*<div className={`tab ${tab === 'link' ? 'active' : ''}`} onClick={() => setTab('link')}>PASTE LINK</div>*/}
         <div className={`tab ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>TEXT</div>
         <div className={`tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>UPLOAD (.txt, .csv)</div>
      </div>

      {/* Input Area: Text input for link/text, File input for upload */}
      <div className="input-row">
        {tab !== 'upload' ? (
          <>
            <input
              type="text"
              placeholder={tab === 'link' ? 'PASTE HERE' : 'Paste or type text...'}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown} // Handle Enter key press
            />
            <button className="add-btn" onClick={handleAddItem}>ADD</button>
          </>
        ) : (
          <input
            type="file"
            onChange={handleFileUpload} // Handle file selection
            accept=".txt,.csv" // Restrict accepted file types
          />
        )}
      </div>

      {/* Items List: Displays the items added by the user */}
      <div className="items">
        {items.map((item, index) => (
          <div key={index} className="item">
             <div className="meta">
               {/* Display item number, type, and value (escaped for safety) */}
               {`${index + 1}. ${item.type.toUpperCase()} - `}
               <span className="muted" dangerouslySetInnerHTML={{ __html: escapeHtml(item.value) }} />
             </div>
             {/* Button to remove the item */}
             <button onClick={() => handleRemoveItem(index)}>Remove</button>
          </div>
        ))}
      </div>

      {/* Analyse Button: Triggers the analysis process */}
      <div className="analyse-row">
        <button
          className="analyse-btn"
          onClick={onAnalyse} // Call the analysis handler passed from App.js
          disabled={loading || items.length === 0} // Disable if loading or no items
        >
          {loading ? 'ANALYSING...' : 'ANALYSE'}
        </button>
      </div>
    </>
  );
}
export default InputPanel;