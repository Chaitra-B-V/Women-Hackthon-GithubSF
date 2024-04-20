from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

video = cv2.VideoCapture("video/Video.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# display_handle = display(None, display_id=True)
# for img in base64Frames:
#     display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
#     time.sleep(0.025)

print("Generating audio description number 1...")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. For this video, take note of key visual elements, actions to describe. Pay attention to details such as colors, shapes, movements, and scene changes. Start describing the video from the beginning, providing an overview of the setting and main characters or objects. Describe any actions or movements taking place in each scene, including gestures, interactions, or changes in the environment. Use descriptive language to convey the mood, atmosphere, and emotions portrayed in the video. Include any relevant audio cues or background sounds that contribute to the overall experience. Structure the narration in a logical sequence, following the chronological order of events in the video. Divide the narration into sections or scenes to facilitate easier navigation and understanding. Ensure that the description is clear, concise, and easy to follow. Provide sufficient detail to paint a vivid picture of the video without overwhelming the listener with unnecessary information. Use sensory language to engage the listener’s imagination and create a rich audio experience.",
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
print(result.choices[0].message.content)
