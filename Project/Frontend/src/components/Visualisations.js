import React from 'react';
// Import necessary components from Chart.js and react-chartjs-2
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

// Register the required Chart.js components globally
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement, Title);

// Visualisations component: Displays doughnut and bar charts based on analysis results.
function Visualisations({ results }) {
    console.log(results);

    // Check if results is a valid array with items
    if (!results || !Array.isArray(results) || results.length === 0) {
        console.log('No results found');
        return null;
    }

    // Helper function to display user-friendly classification names
    const displayClassification = (backendClass) => {
        if (backendClass === 'fake') return 'Misinformation';
        if (backendClass === 'real') return 'Legitimate';
        if (backendClass === 'error') return 'Error';
        return 'Unknown';
    };

    // Count occurrences of each classification ('real', 'fake', 'error', etc.)
    const classificationCounts = {};
    results.forEach(item => {
        const classification = item.classification;
        const displayName = displayClassification(classification);
        classificationCounts[displayName] = (classificationCounts[displayName] || 0) + 1;
    });

    // Check if we have data to display
    if (Object.keys(classificationCounts).length === 0) {
        return <div style={{ padding: '20px', color: '#666' }}>No data available for visualization</div>;
    }

    // Map classification names to colors
    const colorMap = {
        'Legitimate': { bg: 'rgba(75, 192, 192, 0.5)', border: 'rgba(75, 192, 192, 1)' },
        'Misinformation': { bg: 'rgba(255, 99, 132, 0.5)', border: 'rgba(255, 99, 132, 1)' },
        'Error': { bg: 'rgba(201, 203, 207, 0.5)', border: 'rgba(201, 203, 207, 1)' },
        'Unknown': { bg: 'rgba(255, 206, 86, 0.5)', border: 'rgba(255, 206, 86, 1)' }
    };

    const doughnutData = {
        labels: Object.keys(classificationCounts),
        datasets: [
            {
                label: '# of Items',
                data: Object.values(classificationCounts),
                backgroundColor: Object.keys(classificationCounts).map(key => colorMap[key]?.bg || 'rgba(201, 203, 207, 0.5)'),
                borderColor: Object.keys(classificationCounts).map(key => colorMap[key]?.border || 'rgba(201, 203, 207, 1)'),
                borderWidth: 1,
            },
        ],
    };

    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Overall Classification Distribution',
            },
        },
    };

    const barData = {
        labels: results.map((item, index) => `Item ${index + 1} (${item.type || 'N/A'})`),
        datasets: [
            {
                label: 'Confidence Score',
                data: results.map(item => parseFloat(item.confidence)),
                backgroundColor: results.map(item =>
                    item.classification === 'real' ? 'rgba(75, 192, 192, 0.5)' :
                        item.classification === 'fake' ? 'rgba(255, 99, 132, 0.5)' :
                            'rgba(201, 203, 207, 0.5)'
                ),
                borderColor: results.map(item =>
                    item.classification === 'real' ? 'rgba(75, 192, 192, 1)' :
                        item.classification === 'fake' ? 'rgba(255, 99, 132, 1)' :
                            'rgba(201, 203, 207, 1)'
                ),
                borderWidth: 1,
            },
        ],
    };

    // --- Configuration Options for Bar Chart ---
    const barOptions = {
        indexAxis: 'y',
        elements: {
            bar: {
                borderWidth: 2,
            },
        },
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false,
            },
            title: {
                display: true,
                text: 'Confidence Score per Item',
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        if (context.parsed.x !== null) {
                            label += (context.parsed.x);
                        }
                        const itemClassification = results[context.dataIndex].classification;
                        const displayClass = displayClassification(itemClassification);
                        label += ` (${displayClass})`;
                        return label;
                    }
                }
            }
        },
        scales: {
            x: {
                beginAtZero: true,
                max: 100,
                ticks: {
                    callback: function(value) {
                        return (value) ;
                    }
                }
            }
        }
    };

    console.log('Rendering charts with data');

    // Render the container with both charts
    return (
        <div className="visualisations-container" style={{ marginTop: '20px', display: 'flex', gap: '20px', flexWrap: 'wrap', alignItems: 'flex-start' }}>
            {/* Container for the Doughnut chart */}
            <div style={{ flex: '1 1 300px', minWidth: '250px', position: 'relative', height: '300px' }}>
                <Doughnut data={doughnutData} options={doughnutOptions}/>
            </div>
            {/* Container for the Bar chart */}
            <div style={{ flex: '2 1 400px', minWidth: '300px', position: 'relative', height: '300px' }}>
                <Bar options={barOptions} data={barData} />
            </div>
        </div>
    );
}

export default Visualisations;