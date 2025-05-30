from flask_cors import CORS
from flask import Flask, request, jsonify
import openai
import os
import io
from werkzeug.datastructures import FileStorage
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Insurance Call Analyst API is running."

@app.route('/analyze-call', methods=['POST'])
def analyze_call():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    # Handle file upload and convert to BytesIO
    audio_file: FileStorage = request.files['file']
    audio_bytes = io.BytesIO(audio_file.read())
    audio_bytes.name = "call.wav"  # Required for OpenAI

    # Transcribe with Whisper
    transcript_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        language="th"
    )
    transcript = transcript_response.text

    # Generate analysis using ChatGPT
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
