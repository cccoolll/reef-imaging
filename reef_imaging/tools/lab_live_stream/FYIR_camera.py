import os
import cv2
import time
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

# Get the absolute path to the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Camera Index for FYIR Infrared Camera
def gen_frames():
    camera = cv2.VideoCapture("/dev/video0")

    
    if not camera.isOpened():
        logging.error(f"Failed to open the infrared camera")
        return

    # Set manual exposure (adjust based on your environment)
    # camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 for manual mode
    # camera.set(cv2.CAP_PROP_EXPOSURE, -5)  # Adjust between -13 (dark) to -1 (bright)
    # camera.set(cv2.CAP_PROP_GAIN, 10)  # Increase gain if too dark

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logging.error("Failed to capture image")
                break
            else:
                # Convert to grayscale (infrared cameras often work best in grayscale)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Increase brightness if needed
                #bright_frame = cv2.convertScaleAbs(gray_frame, alpha=2, beta=50)  # Adjust alpha/beta as needed

                # Compress the image by adjusting the JPEG quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Adjust quality as needed (0-100)
                ret, buffer = cv2.imencode('.jpg', gray_frame, encode_param)
                if not ret:
                    logging.error("Failed to encode image")
                    break

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)  # Reduce CPU load
    finally:
        camera.release()
        logging.info("Infrared camera resource released")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index_FYIR.html", {"request": request})

@app.get('/video_feed')
async def video_feed(request: Request):
    async def generator():
        for frame in gen_frames():
            if await request.is_disconnected():
                logging.info("Client disconnected, stopping the generator")
                break
            yield frame
    
    return StreamingResponse(generator(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8002)  # Running on a different port