import os
import cv2
import time
import logging
import uvicorn
from fastapi import FastAPI, Request, WebSocket, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

# Get the absolute path to the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Configure logging
logging.basicConfig(level=logging.INFO)

def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logging.error("Failed to open the webcam")
        yield b'--frame\r\n\r\n'  # Return an empty frame to avoid hanging
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                logging.error("Failed to capture image")
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.error("Failed to encode image")
                    break
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    finally:
        camera.release()
        logging.info("Webcam resource released")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/video_feed')
async def video_feed(request: Request):
    async def generator():
        for frame in gen_frames():
            # Check if the client is still connected
            if await request.is_disconnected():
                logging.info("Client disconnected, stopping the generator")
                break
            yield frame
    
    return StreamingResponse(generator(), media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
