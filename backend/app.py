from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import base64
import json
import time
from utils.mediapipe_utils import get_landmarks_with_skeleton
from utils.feature_extractor import extract_all_features
from utils.model import predict_label
from flask_sock import Sock
from flask_cors import CORS

# 加入CORS
app = Flask(__name__)
CORS(app)  # 加入跨域支援
app.config['UPLOAD_FOLDER'] = 'uploads'
sock = Sock(app)
latest_filename = None

@app.route('/api/upload', methods=['POST'])
def upload_video():
    global latest_filename
    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)
    latest_filename = filename  # 記錄最新影片
    return jsonify({"status": "uploaded", "filename": filename})

@sock.route('/api/stream_frames')
def stream_frames(ws):
    global latest_filename
    if latest_filename is None:
        ws.send(json.dumps({"error": "No uploaded video yet"}))
        ws.close()
        return
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], latest_filename)
    if not os.path.exists(filepath):
        ws.send(json.dumps({"error": "file not found"}))
        ws.close()
        return

    cap = cv2.VideoCapture(filepath)
    frame_landmarks = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        landmarks, skeleton_frame = get_landmarks_with_skeleton(frame)
        if landmarks is None:
            continue
        landmarks =  landmarks.landmark
        frame_landmarks.append(landmarks)
        features = extract_all_features(landmarks)

        _, buffer = cv2.imencode('.jpg', skeleton_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        label = predict_label(frame_landmarks)
        ws.send(json.dumps({
            'frame': frame_b64,
            'features': features,
            'result': label
        }))

    cap.release()

    label = predict_label(frame_landmarks)
    ws.send(json.dumps({'result': label}))

    ws.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
