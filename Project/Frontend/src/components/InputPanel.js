// src/components/InputPanel.js
import React, { useState } from 'react';

// This is a helper function from your original app.js to prevent HTML injection
function escapeHtml(str) {
  // ... (the escapeHtml function remains the same)
  return str.replace(/[&<>"']/g, function (m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}

function InputPanel({ tab, setTab, items, setItems, onAnalyse, loading }) {
  const [inputValue, setInputValue] = useState('');

  // --- NEW: Input Validation Logic ---
  const validateInput = (type, value) => {
    if (!value || value.trim() === '') {
      alert('Input cannot be empty.');
      return false;
    }

    // URL validation for the 'link' tab
    if (type === 'link') {
      // A simple regex to check for a valid URL format
      const urlPattern = new RegExp('^[^\\s/$.?#].[^\\s]*$', 'i');
      if (!urlPattern.test(value)) {
        alert('Please enter a valid URL (e.g., http://example.com).');
        return false;
      }
    }
    
    return true; // Input is valid
  };

  const handleAddItem = () => {
    // Use the new validation function before adding
    if (!validateInput(tab, inputValue)) {
      return; // Stop if validation fails
    }

    setItems([...items, { type: tab, value: inputValue.trim() }]);
    setInputValue('');
  };

  const handleRemoveItem = (indexToRemove) => {
    setItems(items.filter((_, index) => index !== indexToRemove));
  };
  
  // --- NEW: File Validation Logic ---
  const handleFileUpload = (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;

    // 1. File Type Validation (allow only text files for this example)
    const allowedTypes = ['text/plain', 'application/pdf', 'text/csv'];
    if (!allowedTypes.includes(file.type)) {
      alert(`Invalid file type. Please upload one of the following: .txt, .pdf, .csv`);
      event.target.value = null; // Reset file input
      return;
    }

    // 2. File Size Validation (limit to 5MB for this example)
    const maxSizeInBytes = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSizeInBytes) {
      alert(`File is too large. Please upload a file smaller than 5MB.`);
      event.target.value = null; // Reset file input
      return;
    }

    setItems([...items, { type: 'upload', value: file.name }]);
    event.target.value = null; 
  };


  return (
    <>
      {/* Tabs */}
      <div className="tabs">
        <div className={`tab ${tab === 'link' ? 'active' : ''}`} onClick={() => setTab('link')}>
          PASTE LINK
        </div>
        <div className={`tab ${tab === 'text' ? 'active' : ''}`} onClick={() => setTab('text')}>
          TEXT
        </div>
        <div className={`tab ${tab === 'upload' ? 'active' : ''}`} onClick={() => setTab('upload')}>
          UPLOAD
        </div>
      </div>

      {/* Input Row */}
      <div className="input-row">
        {tab !== 'upload' ? (
          <>
            <input
              type="text"
              placeholder={tab === 'link' ? 'PASTE HERE' : 'Paste or type text...'}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
            />
            <button className="add-btn" onClick={handleAddItem}>ADD</button>
          </>
        ) : (
          <input type="file" onChange={handleFileUpload} />
        )}
      </div>

      {/* Items List */}
      <div className="items">
        {items.map((item, index) => (
          <div key={index} className="item">
            <div className="meta">
              {`${index + 1}. ${item.type.toUpperCase()} - `}
              <span className="muted" dangerouslySetInnerHTML={{ __html: escapeHtml(item.value) }} />
            </div>
            <button onClick={() => handleRemoveItem(index)}>Remove</button>
          </div>
        ))}
      </div>

      {/* Analyse Row */}
      <div className="analyse-row">
        <button className="analyse-btn" onClick={onAnalyse} disabled={loading || items.length === 0}>
          {loading ? 'ANALYSING...' : 'ANALYSE'}
        </button>
      </div>
    </>
  );
}

export default InputPanel;