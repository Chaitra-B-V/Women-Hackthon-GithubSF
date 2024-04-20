import base64
import logging
import os
import shutil
from typing import Any

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Retrieve the OpenAI API key from environment and create an API client
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

app = FastAPI()

def get_audio_stream(description_text):
    """ Generate an audio stream from the given description text using OpenAI's TTS model. """
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=description_text,
        )
        return response.content if response is not None and hasattr(response, 'content') else None
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    """ Serve an HTML form to upload videos, display the audio player and description text area. """
    return """
    <html>
        <head>
            <title>Upload Video</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Your Website Title</title>
            <link rel="icon" type="image/png" href="logo.png">
        </head>
        <body>
            <h1>Upload Video</h1>
            <form action="/process-video/" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit" value="Upload and Process">
            </form>
            <!-- Audio player for playing the response audio, hidden initially -->
            <audio controls id="audioPlayer" style="display:none;"></audio>
            <!-- Div for displaying description text, hidden initially -->
            <div id="descriptionText" style="display:none;"></div>
            <script>
                const audioPlayer = document.getElementById('audioPlayer');
                const descriptionText = document.getElementById('descriptionText');

                // Add an event listener to handle the form submission
                document.forms[0].onsubmit = async function(event) {
                    event.preventDefault();
                    const formData = new FormData(this);
                    const response = await fetch('/process-video/', {
                        method: 'POST',
                        body: formData,
                    });

                    if (response.ok) {
                        const data = await response.json();
                        audioPlayer.src = 'data:audio/wav;base64,' + data.audio;
                        descriptionText.textContent = data.description;
                        audioPlayer.style.display = 'block';
                        descriptionText.style.display = 'block';
                        audioPlayer.play();
                    } else {
                        alert('Failed to process the video.');
                        audioPlayer.style.display = 'none';
                        descriptionText.style.display = 'none';
                    }
                }
            </script>
        </body>
    </html>
    """

def simulate_openai_api_call(image_b64: str) -> Any:
    """ Simulate an OpenAI API call to process the image and return an audio description. """
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Tell me in one sentence what is happening in this image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                }
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 300,
    }
    result = client.chat.completions.create(**params)
    description_text = result.choices[0].message.content

    # Generate audio from description text
    audio_stream = get_audio_stream(description_text)
    encoded_audio = base64.b64encode(audio_stream).decode('utf-8')
    return {"audio": encoded_audio, "description": description_text}

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """ Endpoint to process the uploaded video, extract a frame, generate description and audio, and return them. """
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File provided is not a video")
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
    video_capture = cv2.VideoCapture(temp_file_path)
    success, frame = video_capture.read()
    video_capture.release()
    os.remove(temp_file_path)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to extract frame from video")
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode('utf-8')
    result = simulate_openai_api_call(image_b64)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
