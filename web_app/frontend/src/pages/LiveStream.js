import React, { useState, useEffect } from 'react';
import { Camera, Activity, AlertTriangle, CheckCircle, WifiOff, Eye, Calendar, UserCheck, X } from 'lucide-react';

const formatTime = (iso) => iso ? new Date(iso).toLocaleTimeString() : "--:--";

export default function LiveStream({ streamUrl, cameraOk, alerts, setCameraOk, cameraRetry, setCameraRetry }) {
  const [selectedDate, setSelectedDate] = useState('');
  const [historyAlerts, setHistoryAlerts] = useState([]);
  
  useEffect(() => {
    if (selectedDate) {
      fetch(`http://localhost:5000/api/logs?date=${selectedDate}`)
        .then(res => res.json())
        .then(data => setHistoryAlerts(data))
        .catch(err => setHistoryAlerts([]));
    }
  }, [selectedDate]);

  const displayAlerts = selectedDate ? historyAlerts : alerts;

  const getImageUrl = (type, identity) => {
    if (type === 'suspect' || type === 'threat') return `http://localhost:5000/blacklist_images/${identity}.jpg`;
    return `http://localhost:5000/api/visitor_image/${identity}`;
  };

  return (
    <div className="live-layout fade-up">
      <div className="camera-panel">
        <div className="panel-header">
          <span className="panel-title"><Camera size={16} color="var(--accent)" /> Smart Gate — Camera 01</span>
          <span className="live-badge"><span className="live-dot" /> LIVE</span>
        </div>
        <div className="camera-feed">
          {cameraOk ? (
            <img
              key={cameraRetry}
              src={streamUrl}
              alt="MJPEG Stream"
              onError={() => {
                setCameraOk(false);
                setTimeout(() => { setCameraOk(true); setCameraRetry(r => r + 1); }, 3000);
              }}
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
            />
          ) : (
            <div className="camera-offline">
              <div className="camera-offline-icon"><WifiOff size={24} /></div>
              <p>Waiting for AI Engine... (Auto-retrying)</p>
            </div>
          )}
        </div>
      </div>

      <div className="alerts-panel">
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={16} color="var(--accent)" /> 
            <span className="panel-title">{selectedDate ? 'Historical Detections' : 'Live Detections'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: 11, color: "var(--text-t)" }}>{displayAlerts.length} Total</span>
            
            {/* التعديل الجديد للـ Date Picker */}
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center', background: 'var(--bg-app)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--border)', transition: 'all 0.3s ease' }}>
              <Calendar size={14} color="var(--accent)" style={{ marginRight: '6px' }} />
              <input 
                type="date" 
                value={selectedDate}
                max={new Date().toISOString().split("T")[0]} // يمنع اختيار تواريخ في المستقبل
                onChange={(e) => setSelectedDate(e.target.value)}
                style={{ background: 'transparent', border: 'none', color: selectedDate ? '#ffffff' : 'var(--text-s)', outline: 'none', fontSize: '13px', cursor: 'pointer', fontFamily: 'inherit' }}
              />
              {selectedDate && (
                <button 
                  onClick={() => setSelectedDate('')} 
                  title="Clear Filter"
                  style={{ background: 'rgba(255, 59, 59, 0.1)', border: 'none', color: 'var(--danger)', cursor: 'pointer', borderRadius: '50%', width: '20px', height: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', marginLeft: '6px', transition: '0.2s' }}
                >
                  <X size={12} />
                </button>
              )}
            </div>
            
          </div>
        </div>
        <div className="alerts-list">
          {displayAlerts.map((a) => {
            const type = a.type || (a.threat_level === "High" ? "suspect" : "new_visitor");
            
            let colorVar, Icon, badgeLabel;
            if (type === 'suspect' || type === 'threat') {
              colorVar = 'var(--danger)';
              Icon = AlertTriangle;
              badgeLabel = 'SUSPECT';
            } else if (type === 'old_visitor') {
              colorVar = '#3b82f6';
              Icon = UserCheck;
              badgeLabel = 'RETURNING VISITOR';
            } else {
              colorVar = 'var(--success)';
              Icon = CheckCircle;
              badgeLabel = 'NEW VISITOR';
            }

            return (
              <div key={a.id} className="alert-item" style={{ borderLeft: `4px solid ${colorVar}`, paddingLeft: '12px' }}>
                <div style={{ width: '40px', height: '40px', borderRadius: '50%', overflow: 'hidden', background: '#000', flexShrink: 0 }}>
                  <img src={getImageUrl(type, a.identity)} alt="face" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.style.display = 'none'; e.target.parentElement.innerHTML = '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:var(--bg-app);color:var(--text-t);font-size:12px;">N/A</div>'; }} />
                </div>
                
                <div className="alert-info" style={{ marginLeft: '12px', flex: 1 }}>
                  <div style={{ color: colorVar, fontWeight: 'bold', fontSize: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Icon size={14} /> {badgeLabel}: {a.identity}
                  </div>
                  <div className="alert-meta" style={{ marginTop: '4px' }}>
                    {formatTime(a.timestamp)} • Confidence Logging
                  </div>
                </div>
              </div>
            );
          })}
          {displayAlerts.length === 0 && <div className="alerts-empty"><Eye size={24} /><span>{selectedDate ? "No logs found for this date" : "Monitoring..."}</span></div>}
        </div>
      </div>
    </div>
  );
}