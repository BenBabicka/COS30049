// src/components/Visualisations.js
import React from 'react';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

// Register the necessary components for Chart.js
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

function Visualisations({ results }) {
  if (!results || !results.items) {
    return null; // Don't render anything if there are no results
  }

  // --- Data for Doughnut Chart ---
  const classificationCounts = results.items.reduce((acc, item) => {
    acc[item.classification] = (acc[item.classification] || 0) + 1;
    return acc;
  }, {});

  const doughnutData = {
    labels: Object.keys(classificationCounts),
    datasets: [
      {
        label: '# of Items',
        data: Object.values(classificationCounts),
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',  // Red for Misinformation (example)
          'rgba(75, 192, 192, 0.5)', // Green for Legitimate (example)
          'rgba(255, 206, 86, 0.5)', // Yellow
          'rgba(54, 162, 235, 0.5)',  // Blue
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(54, 162, 235, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // --- Data for Bar Chart ---
  const barData = {
    labels: results.items.map((item, index) => `Item ${index + 1} (${item.type})`),
    datasets: [
      {
        label: 'Confidence Score',
        data: results.items.map(item => item.confidence),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };
  
  const barOptions = {
    indexAxis: 'y', // Makes the bar chart horizontal
    elements: {
      bar: {
        borderWidth: 2,
      },
    },
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Confidence Score per Item',
      },
    },
    scales: {
        x: {
            beginAtZero: true,
            max: 1 // Assuming confidence is a value between 0 and 1
        }
    }
  };


  return (
    <div className="visualisations-container" style={{ marginTop: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 300px', minWidth: '250px' }}>
        <h4>Overall Classification</h4>
        <Doughnut data={doughnutData} />
      </div>
      <div style={{ flex: '2 1 400px', minWidth: '300px' }}>
        <h4>Individual Analysis</h4>
        <Bar options={barOptions} data={barData} />
      </div>
    </div>
  );
}

export default Visualisations;