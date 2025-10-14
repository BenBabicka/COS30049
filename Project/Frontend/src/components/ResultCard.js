// src/components/ResultCard.js
import React from 'react';

function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str.replace(/[&<>"']/g, function (m) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
  });
}

function ResultCard({ item }) {
  const confidencePercentage = (item.confidence * 100).toFixed(1);
  return (
    <div className="result-card">
      <div>
        <strong>{escapeHtml(item.type.toUpperCase())}</strong>
        <span 
            style={{ 
                backgroundColor: item.classification === 'Misinformation' ? '#fee2e2' : '#dcfce7',
                color: item.classification === 'Misinformation' ? '#991b1b' : '#166534',
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '0.8rem',
                fontWeight: '500',
                marginLeft: '10px'
            }}
        >
            {item.classification}
        </span>
      </div>
      <div className="muted" style={{ wordBreak: 'break-all' }} dangerouslySetInnerHTML={{ __html: escapeHtml(item.value) }} />
      <div style={{ marginTop: '8px' }}>
        Confidence: <strong>{confidencePercentage}%</strong>
      </div>
    </div>
  );
}

export default ResultCard;