@echo off
REM ============================================================
REM  Milestone 4 – MLOps & Monitoring Startup Script
REM  DEPI Facial Recognition – Intelligent Shop Security
REM ============================================================
TITLE Milestone 4 – MLOps ^& Monitoring

echo.
echo  ============================================================
echo   DEPI Facial Recognition - Milestone 4: MLOps and Monitoring
echo  ============================================================
echo.

CALL conda activate AI 2>NUL

cd "C:\Users\Mohsen Hossam\DepiProject\DEPI-Facial-Recognition-Project"

mkdir logs                          2>NUL
mkdir mlflow                        2>NUL
mkdir reports\milestone4_mlops      2>NUL

echo [1/3] Starting MLflow UI (port 5050) ...
START "MLflow UI" cmd /k "conda activate AI && mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5050"
timeout /t 4 /nobreak >NUL

echo [2/3] Starting Monitor API (port 5003) ...
START "Monitor API" cmd /k "conda activate AI && cd /d C:\Users\Mohsen Hossam\DepiProject\DEPI-Facial-Recognition-Project && python mlops\monitoring_scripts\monitor.py --mode api --port 5003 --engine http://localhost:5001"
timeout /t 2 /nobreak >NUL

echo [3/3] Starting MLOps Orchestrator (port 5002) ...
START "MLOps Orchestrator" cmd /k "conda activate AI && cd /d C:\Users\Mohsen Hossam\DepiProject\DEPI-Facial-Recognition-Project && python mlops\monitoring_scripts\mlops_orchestrator.py --port 5002 --engine http://localhost:5001"
timeout /t 2 /nobreak >NUL

echo.
echo  ============================================================
echo   All MLOps services running:
echo    MLflow UI   --> http://localhost:5050
echo    Monitor API --> http://localhost:5003/api/monitor/status
echo    MLOps API   --> http://localhost:5002/api/mlops/status
echo  ============================================================
echo.
echo  Press any key to stop all MLOps services ...
pause >NUL

taskkill /FI "WINDOWTITLE eq MLflow UI"          /F >NUL 2>&1
taskkill /FI "WINDOWTITLE eq Monitor API"         /F >NUL 2>&1
taskkill /FI "WINDOWTITLE eq MLOps Orchestrator"  /F >NUL 2>&1
echo  Stopped. Goodbye!
