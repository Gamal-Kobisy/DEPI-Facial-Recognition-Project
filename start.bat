@echo off
title Intelligent Shop Security
color 0B
chcp 65001 >nul 2>&1

echo.
echo  =========================================================
echo    Intelligent Shop Security System
echo  =========================================================
echo.

:: ==========================================
:: 1. FREE PORTS IF ALREADY IN USE
:: ==========================================
echo  [CLEAN] Freeing ports 3000, 5000, 5001 if busy...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5000 "') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":5001 "') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000 "') do taskkill /F /PID %%a >nul 2>&1
echo  [OK]  Ports cleared
echo.

:: ==========================================
:: 2. BACKEND — Node.js on port 5000
:: ==========================================
echo  [1/3] Starting Backend API Server (Port 5000)...
cd /d "%~dp0web_app\backend"
start "ShopSecurity_Backend" cmd /k "node server.js"
timeout /t 3 /nobreak >nul
echo  [OK]  Backend launched
echo.

:: ==========================================
:: 3. AI ENGINE — Python Flask on port 5001
:: ==========================================
echo  [2/3] Starting AI Face Recognition Engine (Port 5001)...
cd /d "%~dp0ai_engine"
start "ShopSecurity_Camera" cmd /k "conda activate AI && python stream_scanner.py"
timeout /t 5 /nobreak >nul
echo  [OK]  AI Engine launched
echo.

:: ==========================================
:: 4. REACT FRONTEND — port 3000
:: ==========================================
echo  [3/3] Starting React Dashboard (Port 3000)...
cd /d "%~dp0web_app\frontend"
start "ShopSecurity_React" cmd /k "npm start"
echo  [OK]  React launched (Browser will open automatically)
echo.

:: Return to root folder
cd /d "%~dp0"

:: ==========================================
:: READY & SHUTDOWN
:: ==========================================
echo  =========================================================
echo    ✅ All services are running!
echo.
echo    🌐 React UI   -->  http://localhost:3000
echo    📡 Backend    -->  http://localhost:5000
echo    🎥 AI Stream  -->  http://localhost:5001/video_feed
echo.
echo    🛑 Press ANY KEY in this window to STOP everything.
echo  =========================================================
pause