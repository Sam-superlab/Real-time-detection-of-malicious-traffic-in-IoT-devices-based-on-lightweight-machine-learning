from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import yaml
import logging
import threading
from datetime import datetime
from .detector import MaliciousTrafficDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="IoT Malicious Traffic Detection API",
    description="Real-time detection of malicious traffic in IoT devices using lightweight machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = MaliciousTrafficDetector()
detector_thread = None


class DetectionStats(BaseModel):
    """Model for detection statistics."""
    packets_processed: int
    windows_analyzed: int
    malicious_detected: int
    average_processing_time_ms: float
    uptime_seconds: float


class DetectionAlert(BaseModel):
    """Model for detection alerts."""
    timestamp: str
    confidence: float
    source_ips: List[str]
    destination_ips: List[str]
    alert_type: str


# Store recent alerts
recent_alerts: List[DetectionAlert] = []
start_time: Optional[datetime] = None


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "IoT Malicious Traffic Detection API",
        "version": "1.0.0",
        "status": "running" if detector_thread and detector_thread.is_alive() else "stopped"
    }


@app.post("/start")
async def start_detection():
    """Start the detection system."""
    global detector_thread, start_time

    if detector_thread and detector_thread.is_alive():
        raise HTTPException(
            status_code=400, detail="Detection system is already running")

    try:
        detector_thread = threading.Thread(target=detector.start)
        detector_thread.start()
        start_time = datetime.now()
        return {"status": "success", "message": "Detection system started"}
    except Exception as e:
        logger.error(f"Error starting detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop")
async def stop_detection():
    """Stop the detection system."""
    global detector_thread, start_time

    if not detector_thread or not detector_thread.is_alive():
        raise HTTPException(
            status_code=400, detail="Detection system is not running")

    try:
        detector.stop()
        detector_thread.join()
        detector_thread = None
        start_time = None
        return {"status": "success", "message": "Detection system stopped"}
    except Exception as e:
        logger.error(f"Error stopping detection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=DetectionStats)
async def get_stats():
    """Get current detection statistics."""
    if not detector_thread or not detector_thread.is_alive():
        raise HTTPException(
            status_code=400, detail="Detection system is not running")

    uptime = (datetime.now() - start_time).total_seconds() if start_time else 0
    avg_processing_time = (
        sum(detector.stats['processing_time']) /
        len(detector.stats['processing_time']) * 1000
        if detector.stats['processing_time'] else 0
    )

    return DetectionStats(
        packets_processed=detector.stats['packets_processed'],
        windows_analyzed=detector.stats['windows_analyzed'],
        malicious_detected=detector.stats['malicious_detected'],
        average_processing_time_ms=avg_processing_time,
        uptime_seconds=uptime
    )


@app.get("/alerts", response_model=List[DetectionAlert])
async def get_alerts(limit: int = 10):
    """Get recent detection alerts."""
    return recent_alerts[-limit:]


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "window_size": detector.window_size,
        "confidence_threshold": detector.confidence_threshold,
        "interface": detector.interface,
        "feature_list": detector.config['data']['feature_list']
    }


def add_alert(packets: List, probability: float):
    """Add a new alert to the recent alerts list."""
    src_ips = set()
    dst_ips = set()

    for packet in packets:
        if IP in packet:
            src_ips.add(packet[IP].src)
            dst_ips.add(packet[IP].dst)

    alert = DetectionAlert(
        timestamp=datetime.now().isoformat(),
        confidence=probability,
        source_ips=list(src_ips),
        destination_ips=list(dst_ips),
        alert_type="malicious_traffic"
    )

    recent_alerts.append(alert)
    if len(recent_alerts) > 100:  # Keep only last 100 alerts
        recent_alerts.pop(0)


# Override detector's handle_malicious_traffic to add alerts
original_handle = detector.handle_malicious_traffic


def new_handle_malicious_traffic(packets: List, probability: float):
    original_handle(packets, probability)
    add_alert(packets, probability)


detector.handle_malicious_traffic = new_handle_malicious_traffic


def main():
    """Main function to run the API server."""
    import uvicorn

    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        workers=config['api']['workers']
    )


if __name__ == "__main__":
    main()
