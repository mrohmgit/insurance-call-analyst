from flask_cors import CORS
from flask import Flask, request, jsonify
import os
import io
import json
import requests
import time
import tempfile
import ffmpeg
from werkzeug.datastructures import FileStorage
from openai import OpenAI

# Load OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app)

# Load token data
def load_token_data():
    with open("token_data.json", "r") as f:
        return json.load(f)

# Save updated token data
def save_token_data(data):
    with open("token_data.json", "w") as f:
        json.dump(data, f)

# Refresh expired token
def refresh_access_token():
    token_data = load_token_data()
    response = requests.post(
        token_data["token_url"],
        json={
            "grant_type": "refresh_token",
            "refresh_token": token_data["refresh_token"]
        }
    )
    if response.status_code == 200:
        new_token = response.json()
        token_data["access_token"] = new_token["access_token"]
        token_data["access_token_expire_time"] = new_token["access_token_expire_time"]
        save_token_data(token_data)
    else:
        raise Exception("Failed to refresh token")

# Check and refresh if token is expired
def ensure_valid_token():
    token_data = load_token_data()
    current_time = int(time.time())
    if current_time >= token_data["access_token_expire_time"]:
        refresh_access_token()
        token_data = load_token_data()
    return token_data["access_token"], token_data["cdr_url"]

# Downsample audio
def downsample_audio(input_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as original_file:
        original_file.write(input_bytes)
        original_path = original_file.name

    downsampled_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    downsampled_path = downsampled_file.name
    downsampled_file.close()

    try:
        ffmpeg.input(original_path).output(
            downsampled_path,
            ar=16000,
            ac=1,
            format='wav'
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        print("FFmpeg error:", e)
        return None

    return downsampled_path

@app.route('/')
def home():
    return "Insurance Call Analyst API is running."

@app.route('/get-yeastar-calls', methods=['GET'])
def get_yeastar_calls():
    try:
        access_token, cdr_url = ensure_valid_token()

        # Use actual GET parameters from frontend
        start_time = request.args.get("startTime", "2024-05-01T00:00:00")
        end_time = request.args.get("endTime", "2024-05-31T23:59:59")

        payload = {
            "start_time": start_time,
            "end_time": end_time,
            "direction": "inbound",
            "limit": 100,
            "offset": 0
        }

        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.post(cdr_url, json=payload, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch from Yeastar", "status": response.status_code}), 500

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-call', methods=['POST'])
def analyze_call():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file: FileStorage = request.files['file']
    audio_bytes = audio_file.read()

    downsampled_path = downsample_audio(audio_bytes)
    if not downsampled_path:
        return jsonify({'error': 'Audio conversion failed'}), 500

    with open(downsampled_path, "rb") as f:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="th"
        )
    transcript = transcript_response.text

    prompt = f"""
    บทสนทนาเกี่ยวกับการขายประกันรถยนต์:

    \"\"\"{transcript}\"\"\"

    วิเคราะห์บทสนทนา:
    1. จุดแข็งของพนักงานขาย
    2. จุดอ่อน
    3. คำแนะนำที่ควรปรับปรุง

    ตอบเป็นภาษาไทยและเป็นข้อ ๆ
    """

    chat_response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    analysis = chat_response.choices[0].message.content

    return jsonify({
        "transcript": transcript,
        "analysis": analysis
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
