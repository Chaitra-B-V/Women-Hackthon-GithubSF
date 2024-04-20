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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key= openai_api_key)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <head>
            <title>Upload Video1</title>
        </head>
        <body>
            <h1>Upload Video1</h1>
            <form action="/process-video/" enctype="multipart/form-data" method="post">
                <input type="file" name="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """


def simulate_openai_api_call(image_b64: str) -> Any:
    # Placeholder for your OpenAI API call
    # Assume it takes the base64 string of an image and returns some response
    # Replace this with your actual OpenAI API interaction
    # Perform your image processing here (e.g., resizing, cropping, etc.)
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
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
    image_b64 = None
    del image_b64
    #print(result.choices[0].message.content)
    description_text = result.choices[0].message.content
    print(description_text)
    # Convert the description text to audio
    audio_stream = get_audio_stream(description_text)
    # Convert audio_stream to a format that can be sent in the response
    #audio_content = audio_stream.read()  # This depends on how your audiostream is structured
    # Encode the audio content to base64 to send as part of the JSONresponse
    encoded_audio = base64.b64encode(audio_stream).decode('utf-8')
    return encoded_audio
    return {"detail": "Simulated response from OpenAI based on the image."}

def get_audio_stream(description_text):
    try:
        logger.info('get_audio_stream')
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=description_text,
        )
        # Ensure response is successful and contains audio data
        if response is not None and hasattr(response, 'content'):
            # Return the audio content directly
            return response.content
        else:
            print("No audio content in the response")
            return None
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File provided is not a video")

    # Create a temporary file
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(file.file, temp_file)

    # Extract a frame from the video
    video_capture = cv2.VideoCapture(temp_file_path)
    success, frame = video_capture.read()
    video_capture.release()

    # Delete the temporary video file
    os.remove(temp_file_path)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to extract frame from video")

    # Convert the frame to a base64-encoded string
    _, buffer = cv2.imencode('.jpg', frame)
    image_b64 = base64.b64encode(buffer).decode('utf-8')

    # Simulate sending the base64 string to OpenAI
        # Call the function that interacts with OpenAI and gets the base64-encoded audio
    encoded_audio = simulate_openai_api_call(image_b64)

    # Here you might want to overwrite or delete the base64 string if sensitive
    # However, since Python doesn't guarantee immediate memory cleanup, just proceed as best as possible
    frame = None
    image_b64 = None
    del frame
    del image_b64

    # Return the base64-encoded audio in the response
    return JSONResponse(content={"audio": encoded_audio})



