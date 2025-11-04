import React, {useEffect, useState} from 'react';

function ModelStats() {
    const BACKEND_URL = 'http://localhost:8000'; // FastAPI backend URL
    const [stats, setStats] = useState(null);
    useEffect(() => {
        async function fetchData() {
            const response = await fetch(`${BACKEND_URL}/get_stats`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json', // Specify expected response type
                },
            });
            const data = await response.json();
            let dataToDisplay = [];
            for (const key in data) {

                let values = [];
                for (const [k, v] of Object.entries(data[key])) {
                    values.push(<p key={k}>{k}: {v}</p>);
                }
                let newData = <div key={key}><strong>{key}</strong>{values}</div>
                dataToDisplay.push(newData);
            }
            setStats(dataToDisplay);
        }
        fetchData();
    }, []);

    return (
        <>
            <aside className="left-panel card model-stats">
                <h3>MODEL STATS</h3>
                <div>
                    {stats}
                </div>
            </aside>
        </>
    );
}
export default ModelStats;
