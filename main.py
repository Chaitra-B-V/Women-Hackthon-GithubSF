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
from fastapi.staticfiles import StaticFiles
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
app.mount("/static", StaticFiles(directory="static"), name="static")


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
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload Video</title>
    <link rel="icon" type="image/png" href="/static/logo.png">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #6c5ce7;
        }
        form {
            text-align: center;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #fff;
        }
        input[type="submit"] {
            background-color: #fdcb6e;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #f39c12;
        }
        #audioPlayer, #descriptionText {
            margin-top: 20px;
            display: none;
            width: 100%;
        }
        .indent {
            text-indent: 100px; /* Adjust as needed */
        }
    </style>
    </head>
    <body>
    <div class="container">
        <link rel="icon" type="image/jpeg" href="IMG-20240420-WA0001.jpeg">
        <h1>SightPal</h1>
        <p class="indent"><h3>What do you see?</h3></p>
        <form action="/process-video/" enctype="multipart/form-data" method="post">
            <input type="file" name="file"><br>
            <input type="submit" value="Upload and Process">
        </form>
        <audio controls id="audioPlayer"></audio>
        <div id="descriptionText"></div>
    </div>

    <script>
        const audioPlayer = document.getElementById('audioPlayer');
        const descriptionText = document.getElementById('descriptionText');

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
    

def simulate_openai_api_call(base64Frames: list) -> Any:
    """ Simulate an OpenAI API call to process the image and return an audio description. """
    PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. Create a video description for a visually impaired person. Specifically, I want to know if there's anything of interest that would affect the visually impaired person's decision making on safety. I don't need the full description of everything going on in the environment, but I need to be alerted if there's a concern, and reassured if not.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::60]),
        ],
    },
]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
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
    video = cv2.VideoCapture(temp_file_path)
    
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    result = simulate_openai_api_call(base64Frames)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
