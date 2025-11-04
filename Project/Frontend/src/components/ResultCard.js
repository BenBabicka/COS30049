// src/components/ResultCard.js
import React from 'react';

// Simple function to escape basic HTML characters to prevent XSS issues when rendering input values
function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str.replace(/[&<>"']/g, function (m) {
     // Map special characters to their HTML entity equivalents
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}

// ResultCard component: Displays a single analysis result item.
function ResultCard({ item }) {
  // Calculate confidence percentage and format to one decimal place
  const confidencePercentage = item['confidence'];
  // Get the display text for the current item's classification
  const itemDisplayClassification = item['classification'];

  // Determine background and text colors based on classification for visual indication
  let bgColor, textColor;
  switch (item['classification']) {
    case 'fake':
      bgColor = '#fee2e2'; // Light red background
      textColor = '#991b1b'; // Dark red text
      break;
    case 'real':
      bgColor = '#dcfce7'; // Light green background
      textColor = '#166534'; // Dark green text
      break;
    case 'error':
      bgColor = '#f3f4f6'; // Light gray background
      textColor = '#4b5563'; // Dark gray text
      break;
    default:
      bgColor = '#fef3c7'; // Light yellow background for unknown
      textColor = '#92400e'; // Dark yellow/brown text for unknown
  }


  return (
    <div className="result-card">
      {/* Card Header: Displays item type and classification badge */}
      <div>
        {/* Display the item type (e.g., TEXT, LINK), with a fallback */}
        <strong>{escapeHtml(item['type'])}</strong>
        {/* Badge showing the classification (Misinformation/Legitimate/Error) with appropriate colors */}
        <span
            style={{
                backgroundColor: bgColor,
                color: textColor,
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '0.8rem',
                fontWeight: '500',
                marginLeft: '10px'
            }}
        >
            {itemDisplayClassification}
        </span>
      </div>

      {/* Display the original input value (escaped for safety) */}
      {/* 'wordBreak: break-all' prevents long URLs/text from overflowing */}
      <div className="muted" style={{ wordBreak: 'break-all', marginTop: '4px' }} dangerouslySetInnerHTML={{ __html: escapeHtml(item['tweet']) }} />

      {/* Display the confidence score percentage */}
      <div style={{ marginTop: '8px' }}>
        Confidence: <strong>{confidencePercentage}</strong>
      </div>
    </div>
  );
}

export default ResultCard;