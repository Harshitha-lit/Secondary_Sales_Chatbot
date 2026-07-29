import React, { useState, useMemo } from 'react';
import './index.css';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [hasPredicted, setHasPredicted] = useState(false);

  const handlePredict = () => {
    setLoading(true);
    setError(null);
    fetch('http://localhost:8001/api/secondary-predictions')
      .then(res => res.json())
      .then(result => {
        if (result.status === 'success') {
          setData(result.data);
          setHasPredicted(true);
        } else {
          setError(result.error || 'Failed to fetch data');
        }
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  };

  const filteredData = useMemo(() => {
    if (filterStatus === 'ALL') return data;
    return data.filter(item => item.status === filterStatus);
  }, [data, filterStatus]);

  // Calculations for summary cards
  const totalRisk = filteredData.reduce((acc, curr) => acc + (curr.value_at_risk || 0), 0);
  const lapsedCount = data.filter(item => item.status === 'Lapsed').length;
  const decliningCount = data.filter(item => item.status === 'Declining').length;
  const healthyCount = data.filter(item => item.status === 'Healthy').length;

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(val);
  };

  const getBadgeClass = (status) => {
    if (status === 'Healthy') return 'badge healthy';
    if (status === 'Declining') return 'badge declining';
    if (status === 'Lapsed') return 'badge lapsed';
    return 'badge';
  };
  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Secondary Data Churn Risk Dashboard</h1>
        <p>Retailer grain activity and financial risk predictions.</p>
        {!hasPredicted && !loading && (
          <button 
            onClick={handlePredict} 
            className="predict-btn"
            style={{
              marginTop: '1.5rem',
              padding: '0.75rem 1.5rem',
              backgroundColor: 'var(--primary, #4F46E5)',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: 'pointer',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
          >
            Predict Churn Risk
          </button>
        )}
      </div>

      {loading && <div className="loading">Loading risk predictions...</div>}
      {error && <div className="error">Error: {error}</div>}

      {hasPredicted && !loading && !error && (
        <>
          <div className="cards-grid">
        <div 
          className={`summary-card ${filterStatus === 'ALL' ? 'active' : ''}`}
          onClick={() => setFilterStatus('ALL')}
        >
          <div className="card-title">Filtered Value at Risk</div>
          <div className="card-value risk">{formatCurrency(totalRisk)}</div>
        </div>
        
        <div 
          className={`summary-card ${filterStatus === 'Lapsed' ? 'active' : ''}`}
          onClick={() => setFilterStatus('Lapsed')}
        >
          <div className="card-title">Lapsed Accounts</div>
          <div className="card-value">{lapsedCount}</div>
        </div>
        
        <div 
          className={`summary-card ${filterStatus === 'Declining' ? 'active' : ''}`}
          onClick={() => setFilterStatus('Declining')}
        >
          <div className="card-title">Declining Accounts</div>
          <div className="card-value">{decliningCount}</div>
        </div>

        <div 
          className={`summary-card ${filterStatus === 'Healthy' ? 'active' : ''}`}
          onClick={() => setFilterStatus('Healthy')}
        >
          <div className="card-title">Healthy Accounts</div>
          <div className="card-value">{healthyCount}</div>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Retailer ID (SK)</th>
              <th>SKU (SK)</th>
              <th>Status</th>
              <th>Churn Prob.</th>
              <th>Months Inactive</th>
              <th>Value at Risk (₹)</th>
            </tr>
          </thead>
          <tbody>
            {filteredData.slice(0, 100).map((row, idx) => (
              <tr key={idx}>
                <td>{row.outlet_sk}</td>
                <td>{row.sku_sk}</td>
                <td>
                  <span className={getBadgeClass(row.status)}>{row.status}</span>
                </td>
                <td>{(row.churn_probability * 100).toFixed(1)}%</td>
                <td>{row.months_inactive}</td>
                <td>{formatCurrency(row.value_at_risk)}</td>
              </tr>
            ))}
            {filteredData.length === 0 && (
              <tr>
                <td colSpan="6" style={{textAlign: 'center'}}>No accounts match this filter.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
        </>
      )}
    </div>
  );
}

export default App;
