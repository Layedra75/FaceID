import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_frame_from_url(url):
    response = requests.get(url)
    return np.asarray(bytearray(response.content), dtype="uint8")

def extract_and_match_frames(img_url, vid_url, max_frames=50):
    frame = cv2.imdecode(get_frame_from_url(img_url), cv2.IMREAD_GRAYSCALE)
    _, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cap = cv2.VideoCapture(vid_url)
    
    if not cap.isOpened():
        return "Error: Cannot open video file."

    frames_processed = 0

    while frames_processed < max_frames:
        ret, vid_frame = cap.read()

        if not ret:
            break

        vid_frame_gray = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
        vid_frame_gray = cv2.resize(vid_frame_gray, (frame.shape[1], frame.shape[0]))

        # Comparación de correlación normalizada
        correlation = cv2.matchTemplate(frame, vid_frame_gray, cv2.TM_CCOEFF_NORMED)

        # Obtenemos la posición del valor máximo
        min_val, max_val, _, _ = cv2.minMaxLoc(correlation)

        # Puedes ajustar este umbral según tus necesidades
        similarity_threshold = 0.7

        if max_val > similarity_threshold:
            cap.release()
            result = {"similarity": max_val, "match_status": "Coinciden", "image_url": img_url, "video_url": vid_url}
            return result

        frames_processed += 1

    cap.release()
    result = {"similarity": 0, "match_status": "No coinciden", "image_url": img_url, "video_url": vid_url}
    return result

@app.route('/', methods=['POST'])
def process_request():
    img_url = request.json['image_url']
    vid_url = request.json['video_url']
    return jsonify(extract_and_match_frames(img_url, vid_url))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)