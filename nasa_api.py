import requests
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

NASA_KEY = os.getenv("NASA_KEY")

def fetch_nasa_image():
    url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_KEY}"

    data = requests.get(url).json()

    img_url = data["url"]
    img_bytes = requests.get(img_url).content

    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    return img
