import os
import cv2
import time
import logging
import uvicorn
import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from threading import Thread, Event

# Get the absolute path to the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Change from a global camera to a function that returns a fresh camera object
def get_camera():
    cam = cv2.VideoCapture(4)
    # Force camera settings refresh
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cam

recording_event = Event()
recording_thread = None
frame_bytes = None

def gen_frames(camera_instance):
    global frame_bytes
    while True:
        success, frame = camera_instance.read()
        if not success:
            logging.error("Failed to capture image")
            frame_bytes = None  # Clear frame_bytes on error
            break
        else:
            # Convert to grayscale (infrared cameras often work best in grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rotate 180 degrees
            gray_frame = cv2.rotate(gray_frame, cv2.ROTATE_180)
            # Add timestamp to the frame
            timestamp = time.strftime('%H:%M', time.localtime())
            cv2.putText(gray_frame, timestamp, (gray_frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Compress the image by adjusting the JPEG quality
            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Adjust quality as needed (0-100)
            ret, buffer = cv2.imencode('.jpg', gray_frame) #, encode_param)
            if not ret:
                logging.error("Failed to encode image")
                frame_bytes = None  # Clear frame_bytes on error
                break

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.1)  # Reduce CPU load

def record_time_lapse():
    global frame_bytes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    filename = f"time_lapse_{timestamp}.mp4"
    out = cv2.VideoWriter(os.path.join(base_dir, "static", filename), fourcc, 24, (640, 480))

    start_time = time.time()
    duration = 1 * 60 * 60  # 1 hour
    interval = 1 / 24 * 60  # 60x speed up

    while time.time() - start_time < duration:
        if recording_event.is_set():
            if frame_bytes:
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                out.write(frame)
            else:
                camera = get_camera()
                success, frame = camera.read()
                #compress the image by adjusting the JPEG quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                if not ret:
                    logging.error("Failed to encode image")
                    break
                frame_bytes = buffer.tobytes()
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                #add timestamp to the frame
                timestamp = time.strftime('%H:%M', time.localtime())
                cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                if success:
                    out.write(frame)
                camera.release()
            time.sleep(interval)
        else:
            break

    out.release()
    logging.info("Time-lapse recording finished")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
async def video_feed(request: Request):
    global frame_bytes
    frame_bytes = None  # Clear the previous frame
    
    # Get a fresh camera instance for each connection
    camera_instance = get_camera()
    
    async def generator():
        try:
            for frame in gen_frames(camera_instance):
                if await request.is_disconnected():
                    logging.info("Client disconnected, stopping the generator")
                    frame_bytes = None
                    break
                yield frame
        finally:
            # Make sure to release the camera when done
            camera_instance.release()
            logging.info("Camera released")
    
    return StreamingResponse(generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post('/start_time_lapse')
async def start_time_lapse(request: Request):
    global recording_thread
    if recording_event.is_set():
        return JSONResponse(content={"message": "Recording in progress"}, status_code=400)
    else:
        recording_event.set()
        recording_thread = Thread(target=record_time_lapse)
        recording_thread.start()
        return JSONResponse(content={"message": "Time-lapse recording started"}, status_code=200)

@app.post('/stop_time_lapse')
async def stop_time_lapse(request: Request):
    recording_event.clear()
    if recording_thread:
        recording_thread.join()
    return JSONResponse(content={"message": "Time-lapse recording stopped"}, status_code=200)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)  # Running on a different port