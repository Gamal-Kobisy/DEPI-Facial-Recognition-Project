import React from "react";
import { ShieldCheck, Cpu, Camera, Lock } from "lucide-react";
import "./LoadingScreen.css";

export default function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-container">
        
        {/* Logo Animation */}
        <div className="loading-logo-wrapper">
          <div className="scan-line"></div>
          <ShieldCheck size={50} className="main-icon" />
        </div>

        {/* Text */}
        <div className="loading-text">
          <h2>Intelligent <span>Security</span></h2>
          <p>Initializing AI Engine & Camera Feeds...</p>
        </div>

        {/* Progress Bar */}
        <div className="loading-progress">
          <div className="progress-bar">
            <div className="progress-fill"></div>
          </div>
        </div>

        {/* Badges */}
        <div className="loading-badges">
          <div className="badge">
            <Camera size={20} />
            <span>Vision API</span>
          </div>
          <div className="badge">
            <Cpu size={20} />
            <span>AI Models</span>
          </div>
          <div className="badge">
            <Lock size={20} />
            <span>Encrypted</span>
          </div>
        </div>

      </div>
    </div>
  );
}