import React, { useState, useEffect, useRef } from "react";
import { Camera, ShieldAlert, BarChart3, Users } from "lucide-react";
import { io } from "socket.io-client";
import "./App.css";
import LoadingScreen from "./components/LoadingScreen";
import LiveStream from "./pages/LiveStream";
import Blacklist from "./pages/Blacklist";
import Dashboard from "./pages/Dashboard";
import Visitors from "./pages/Visitors";

const BACKEND_URL = "http://localhost:5000";
const STREAM_URL  = "http://localhost:5001/video_feed";

const NAV_ITEMS = [
  { id: "live",      label: "Live Stream",  Icon: Camera },
  { id: "visitors",  label: "Visitors",     Icon: Users },
  { id: "blacklist", label: "Blacklist",    Icon: ShieldAlert },
  { id: "dashboard", label: "Dashboard",    Icon: BarChart3 },
];

const playSystemAlert = () => {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return;
  
  const ctx = new AudioContext();
  
  const createBeep = (startTime, freq) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    
    osc.type = 'square';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, startTime + 0.3);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start(startTime);
    osc.stop(startTime + 0.3);
  };

  createBeep(ctx.currentTime, 800); 
  createBeep(ctx.currentTime + 0.2, 1000); 
};

export default function ShopSecurityApp() {
  const [loading, setLoading]       = useState(true);
  const [activeTab, setActiveTab]   = useState("live");
  const [alerts, setAlerts]         = useState([]);
  const [isConnected, setConnected] = useState(false);
  const [cameraOk, setCameraOk]     = useState(true);
  const [cameraRetry, setCameraRetry] = useState(0);
  const socketRef = useRef(null);

  useEffect(() => {
    const timer = setTimeout(() => setLoading(false), 2500);

    fetch(`${BACKEND_URL}/api/logs`)
      .then(r => r.json())
      .then(setAlerts)
      .catch(() => console.log("Backend offline"));
    
    socketRef.current = io(BACKEND_URL);
    socketRef.current.on("connect", () => setConnected(true));
    socketRef.current.on("disconnect", () => setConnected(false));
    
    socketRef.current.on("security_alert", (alert) => {
      setAlerts((prev) => [alert, ...prev].slice(0, 50));
      
      if(alert.threat_level === "High") {
          try { playSystemAlert(); } catch (e) { console.log("Audio blocked by browser."); }
      }
    });

    return () => {
      clearTimeout(timer);
      socketRef.current?.disconnect();
    };
  }, []);

  if (loading) return <LoadingScreen />;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        {/* Logo */}
        <div className="logo">
          <div className="logo-icon"><ShieldAlert size={18} /></div>
          <div className="logo-text" style={{fontSize: '14px'}}>Intelligent<span>Security</span></div>
        </div>
        
        <div className="nav-section">Navigation</div>
        <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {NAV_ITEMS.map(({ id, label, Icon }) => (
            <button key={id} className={`nav-btn ${activeTab === id ? "active" : ""}`} onClick={() => setActiveTab(id)}>
              <Icon size={16} /> {label}
            </button>
          ))}
        </nav>
        
        <div className="sidebar-footer">
          <div className="system-status">
            <span className={`status-dot ${isConnected ? "" : "offline"}`} />
            <span style={{ fontSize: 12, color: "var(--text-s)" }}>{isConnected ? "Backend connected" : "Offline"}</span>
          </div>
          <div className="system-status" style={{ marginTop: 8 }}>
            <span className={`status-dot ${cameraOk ? "" : "offline"}`} />
            <span style={{ fontSize: 12, color: "var(--text-s)" }}>{cameraOk ? "Camera live" : "Camera offline"}</span>
          </div>
        </div>
      </aside>

      <div className="main-content">
        <div className="page-header">
          <div className="page-title">
            {activeTab === "live" && "Live Stream"}
            {activeTab === "visitors" && "Visitors Management"}
            {activeTab === "blacklist" && "Blacklist Management"}
            {activeTab === "dashboard" && "System Dashboard"}
          </div>
        </div>
        
        <div className="page-body">
          {activeTab === "live" && <LiveStream streamUrl={STREAM_URL} cameraOk={cameraOk} alerts={alerts} setCameraOk={setCameraOk} cameraRetry={cameraRetry} setCameraRetry={setCameraRetry} />}
          {activeTab === "visitors" && <Visitors />}
          {activeTab === "blacklist" && <Blacklist />}
          {activeTab === "dashboard" && <Dashboard alerts={alerts} />}
        </div>
      </div>
    </div>
  );
}