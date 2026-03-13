import React from 'react';
import { Camera, Activity, AlertTriangle, CheckCircle, WifiOff, Eye } from 'lucide-react';

const formatTime = (iso) => iso ? new Date(iso).toLocaleTimeString() : "--:--";

export default function LiveStream({ streamUrl, cameraOk, alerts, setCameraOk, cameraRetry, setCameraRetry }) {
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
        <div className="panel-header">
          <span className="panel-title"><Activity size={16} color="var(--accent)" /> Live Detections</span>
          <span style={{ fontSize: 11, color: "var(--text-t)" }}>{alerts.length} Total</span>
        </div>
        <div className="alerts-list">
          {alerts.map((a) => {

            const isThreat = a.threat_level === "High";
            
            return (
              <div key={a.id} className={`alert-item ${isThreat ? "threat" : "safe"}`}>
                <div className={`alert-icon ${isThreat ? "threat" : "safe"}`}>
                  {isThreat ? <AlertTriangle size={15} /> : <CheckCircle size={15} />}
                </div>
                <div className="alert-info">
                  {/* Name & Meta */}
                  <div className={`alert-name ${isThreat ? "threat" : "safe"}`}>
                    {isThreat ? `🚨 SUSPECT: ${a.identity}` : `✅ ${a.identity}`}
                  </div>
                  <div className="alert-meta">
                    {isThreat ? `Threat: ${a.threat_level}` : "Safe Visitor"} • {formatTime(a.timestamp)}
                  </div>
                </div>
              </div>
            );
          })}
          {alerts.length === 0 && <div className="alerts-empty"><Eye size={24} /><span>Monitoring...</span></div>}
        </div>
      </div>
    </div>
  );
}