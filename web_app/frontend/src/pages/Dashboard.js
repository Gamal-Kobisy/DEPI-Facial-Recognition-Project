import React, { useState } from 'react';
import { BarChart3, Clock, Users, ShieldAlert, Filter } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const formatTime = (iso) => iso ? new Date(iso).toLocaleTimeString() : "--:--";
const formatDate = (iso) => iso ? new Date(iso).toLocaleDateString() : "--/--/----";

export default function Dashboard({ alerts }) {
  const [timeFilter, setTimeFilter] = useState('today');

  const now = new Date();
  const filteredAlerts = alerts.filter(a => {
    const logDate = new Date(a.timestamp);
    if (timeFilter === 'today') return logDate.toDateString() === now.toDateString();
    if (timeFilter === 'week') {
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(now.getDate() - 7);
      return logDate >= oneWeekAgo;
    }
    if (timeFilter === 'month') {
      const oneMonthAgo = new Date();
      oneMonthAgo.setMonth(now.getMonth() - 1);
      return logDate >= oneMonthAgo;
    }
    return true;
  });

  const totalVisitors = filteredAlerts.filter(a => a.threat_level === "Safe").length;
  const totalThreats = filteredAlerts.filter(a => a.threat_level === "High").length;

  const pieData = [
    { name: 'Safe Visitors', value: totalVisitors },
    { name: 'Security Threats', value: totalThreats }
  ];
  const COLORS = ['#00f2c3', '#ff4d6d'];

  const chartDataMap = {};
  
  filteredAlerts.forEach(log => {
    const d = new Date(log.timestamp);
    let keyStr = "";
    
    if (timeFilter === 'today') {
      keyStr = d.toLocaleTimeString('en-US', { hour: 'numeric', hour12: true });
    } 

    else {
      keyStr = d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }

    if (!chartDataMap[keyStr]) {
      chartDataMap[keyStr] = { label: keyStr, Visitors: 0, Threats: 0, _time: d.getTime() };
    }

    if (log.threat_level === "Safe") chartDataMap[keyStr].Visitors += 1;
    if (log.threat_level === "High") chartDataMap[keyStr].Threats += 1;
  });

  const barData = Object.values(chartDataMap).sort((a, b) => a._time - b._time);

  return (
    <div className="fade-up">
      {/* time filter */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}><BarChart3 color="var(--accent)"/> System Analytics</h2>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'var(--bg-div)', padding: '8px 15px', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <Filter size={16} color="var(--text-s)" />
          <select 
            value={timeFilter} 
            onChange={(e) => setTimeFilter(e.target.value)}
            style={{ background: 'var(--bg-div)', border: 'none', color: 'var(--text-p)', outline: 'none', cursor: 'pointer', fontSize: '14px', fontWeight: 'bold' }}
          >
            <option value="today" style={{ background: '#1e1e2f', color: '#ffffff' }}>Today's Data (Hourly)</option>
            <option value="week" style={{ background: '#1e1e2f', color: '#ffffff' }}>Last 7 Days</option>
            <option value="month" style={{ background: '#1e1e2f', color: '#ffffff' }}>This Month</option>
            <option value="all" style={{ background: '#1e1e2f', color: '#ffffff' }}>All Time History</option>
          </select>
        </div>
      </div>

      {/* KPIs */}
      <div className="stats-grid">
        <div className="stat-card" style={{ borderBottom: '4px solid var(--success)' }}>
          <div className="stat-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
            TOTAL VISITORS <Users size={18} color="var(--success)"/>
          </div>
          <div className="stat-value green" style={{ fontSize: '36px', marginTop: '10px' }}>{totalVisitors}</div>
        </div>
        
        <div className="stat-card" style={{ borderBottom: '4px solid var(--danger)' }}>
          <div className="stat-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
            SECURITY THREATS <ShieldAlert size={18} color="var(--danger)"/>
          </div>
          <div className="stat-value red" style={{ fontSize: '36px', marginTop: '10px' }}>{totalThreats}</div>
        </div>
      </div>

      {/* graphs */}
      <div style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
        {/* Bar Chart */}
        <div style={{ flex: 2, background: 'var(--bg-div)', borderRadius: '16px', padding: '20px', border: '1px solid var(--border)' }}>
          <h3 style={{ margin: '0 0 20px 0', fontSize: '14px', color: 'var(--text-s)', textTransform: 'uppercase' }}>
            TRAFFIC & THREATS ({timeFilter === 'today' ? 'HOURLY' : 'DAILY'})
          </h3>
          <div style={{ height: '250px', width: '100%' }}>
            {barData.length > 0 ? (
              <ResponsiveContainer>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="label" stroke="var(--text-s)" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="var(--text-s)" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                  <RechartsTooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor: 'var(--bg-card)', borderColor: 'var(--border)', color: '#fff', borderRadius: '8px'}} />
                  <Bar dataKey="Visitors" fill="var(--success)" radius={[4, 4, 0, 0]} barSize={20} />
                  <Bar dataKey="Threats" fill="var(--danger)" radius={[4, 4, 0, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-s)' }}>No data available for this period.</div>
            )}
          </div>
        </div>

        {/* Pie Chart */}
        <div style={{ flex: 1, background: 'var(--bg-div)', borderRadius: '16px', padding: '20px', border: '1px solid var(--border)' }}>
          <h3 style={{ margin: '0 0 20px 0', fontSize: '14px', color: 'var(--text-s)' }}>SAFETY RATIO</h3>
          <div style={{ height: '250px', width: '100%' }}>
            {totalVisitors > 0 || totalThreats > 0 ? (
               <ResponsiveContainer>
                 <PieChart>
                   <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value" stroke="none">
                     {pieData.map((entry, index) => <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />)}
                   </Pie>
                   <RechartsTooltip contentStyle={{backgroundColor: 'var(--bg-card)', borderColor: 'var(--border)', color: '#fff', borderRadius: '8px'}} />
                 </PieChart>
               </ResponsiveContainer>
            ) : (
               <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-s)' }}>No traffic.</div>
            )}
          </div>
        </div>
      </div>

      {/* the logs panel */}
      <div className="logs-panel">
        <div className="panel-header" style={{ marginBottom: 20 }}>
          <span className="panel-title"><Clock size={16} color="var(--accent)" /> Detailed Logs ({timeFilter.toUpperCase()})</span>
        </div>
        <table className="logs-table">
          <thead><tr><th>Identity</th><th>Status</th><th>Date</th><th>Time</th></tr></thead>
          <tbody>
            {filteredAlerts.length === 0 ? (
              <tr><td colSpan={4} style={{ textAlign: "center", color: "var(--text-t)", padding: "40px" }}>No records found for the selected time filter.</td></tr>
            ) : (
              filteredAlerts.slice(0, 30).map((a) => (
                <tr key={a.id}>
                  <td style={{ color: "var(--text-p)", fontWeight: 700 }}>{a.identity}</td>
                  <td>
                    {a.threat_level === "High" 
                      ? <span className="threat-badge high">Threat</span> 
                      : <span className="threat-badge" style={{background: 'rgba(0, 242, 195, 0.1)', color: 'var(--success)', border: '1px solid rgba(0, 242, 195, 0.2)'}}>Safe Visitor</span>
                    }
                  </td>
                  <td>{formatDate(a.timestamp)}</td>
                  <td>{formatTime(a.timestamp)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}