from flask_cors import CORS
from flask import Flask, request, jsonify
import openai
import os
import io
import ffmpeg
import tempfile
from werkzeug.datastructures import FileStorage
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Function to downsample audio to 16kHz mono WAV
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

@app.route('/analyze-call', methods=['POST'])
def analyze_call():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file: FileStorage = request.files['file']
    audio_bytes = audio_file.read()

    # Downsample the audio
    downsampled_path = downsample_audio(audio_bytes)
    if not downsampled_path:
        return jsonify({'error': 'Audio conversion failed'}), 500

    # Transcribe audio using Whisper
    with open(downsampled_path, "rb") as f:
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="th"
        )
    transcript = transcript_response.text

    # Prompt for GPT analysis
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
