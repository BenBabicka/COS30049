// src/components/Visualisations.js
import React from 'react';
// Import necessary components from Chart.js and react-chartjs-2
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

// Register the required Chart.js components globally
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

// Visualisations component: Displays doughnut and bar charts based on analysis results.
function Visualisations({ results }) {
  // Don't render anything if results or result items are not available
  if (!results || !results.items) {
    return null;
  }

  // --- Data Preparation for Doughnut Chart (Overall Classification) ---
  // Count occurrences of each classification ('real', 'fake', 'error', etc.)
  const classificationCounts = results.items.reduce((acc, item) => {
    // Map backend 'real'/'fake' to display labels
    const label = item.classification === 'real' ? 'Legitimate' :
                  item.classification === 'fake' ? 'Misinformation' :
                  item.classification === 'error' ? 'Error' : 'Unknown';
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {});

  const doughnutData = {
    labels: Object.keys(classificationCounts), // Labels for each slice (Legitimate, Misinformation, Error)
    datasets: [
      {
        label: '# of Items', // Dataset label for tooltips
        data: Object.values(classificationCounts), // Data values for each slice
        // Define background colors for each classification type
        backgroundColor: [
          'rgba(75, 192, 192, 0.5)',  // Green for Legitimate
          'rgba(255, 99, 132, 0.5)',   // Red for Misinformation
          'rgba(201, 203, 207, 0.5)', // Grey for Error
          'rgba(255, 206, 86, 0.5)',  // Yellow for Unknown or others
          // Add more colors if more classifications are possible
        ],
        // Define border colors for each slice
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(201, 203, 207, 1)',
          'rgba(255, 206, 86, 1)',
          // Add corresponding border colors
        ],
        borderWidth: 1, // Border width for slices
      },
    ],
  };

   const doughnutOptions = {
    plugins: {
      legend: {
        position: 'top', // Position the legend at the top
      },
      title: {
        display: true,
        text: 'Overall Classification Distribution', // Chart title
      },
    },
  };


  // --- Data Preparation for Bar Chart (Confidence Scores per Item) ---
  const barData = {
    // Labels for the Y-axis (Item number and type)
    labels: results.items.map((item, index) => `Item ${index + 1} (${item.type?.toUpperCase() || 'N/A'})`),
    datasets: [
      {
        label: 'Confidence Score', // Dataset label
        // Data for the X-axis (confidence scores)
        data: results.items.map(item => item.confidence),
        // Assign bar color based on classification
        backgroundColor: results.items.map(item =>
            item.classification === 'real' ? 'rgba(75, 192, 192, 0.5)' : // Green for Legitimate
            item.classification === 'fake' ? 'rgba(255, 99, 132, 0.5)' : // Red for Misinformation
            'rgba(201, 203, 207, 0.5)' // Grey for Error/Unknown
        ),
        borderColor: results.items.map(item =>
            item.classification === 'real' ? 'rgba(75, 192, 192, 1)' :
            item.classification === 'fake' ? 'rgba(255, 99, 132, 1)' :
            'rgba(201, 203, 207, 1)'
        ),
        borderWidth: 1, // Border width for bars
      },
    ],
  };

  // --- Configuration Options for Bar Chart ---
  const barOptions = {
    indexAxis: 'y', // Makes the bar chart horizontal for better readability of item labels
    elements: {
      bar: {
        borderWidth: 2, // Slightly thicker border for bars
      },
    },
    responsive: true, // Make the chart responsive to container size
    maintainAspectRatio: false, // Allow chart aspect ratio to change
    plugins: {
      legend: {
        display: false, // Hide legend as color indicates classification
      },
      title: {
        display: true, // Display the chart title
        text: 'Confidence Score per Item', // Title text
      },
       tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.x !== null) {
              // Format confidence as percentage in tooltip
              label += (context.parsed.x * 100).toFixed(1) + '%';
            }
            // Add classification to tooltip
            const itemClassification = results.items[context.dataIndex]?.classification;
            const displayClass = displayClassification(itemClassification); // Use helper function
            label += ` (${displayClass})`;
            return label;
          }
        }
      }
    },
    scales: {
        x: { // Configuration for the X-axis (Confidence Score)
            beginAtZero: true, // Start axis at 0
            max: 1, // Set maximum value to 1 (since confidence is 0-1)
             ticks: {
                 callback: function(value) {
                     return (value * 100) + '%'; // Format ticks as percentages
                 }
             }
        }
    }
  };

  // Helper function for tooltip label
  const displayClassification = (backendClass) => {
    if (backendClass === 'fake') return 'Misinformation';
    if (backendClass === 'real') return 'Legitimate';
    if (backendClass === 'error') return 'Error';
    return 'Unknown';
  };


  // Render the container with both charts
  return (
    <div className="visualisations-container" style={{ marginTop: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
      {/* Container for the Doughnut chart */}
      <div style={{ flex: '1 1 300px', minWidth: '250px', position: 'relative', height: '300px' }}>
        {/*<h4>Overall Classification</h4>*/}
        <Doughnut data={doughnutData} options={doughnutOptions}/>
      </div>
      {/* Container for the Bar chart */}
      <div style={{ flex: '2 1 400px', minWidth: '300px', position: 'relative', height: '300px' }}>
        {/*<h4>Individual Analysis</h4>*/}
        <Bar options={barOptions} data={barData} />
      </div>
    </div>
  );
}

export default Visualisations;