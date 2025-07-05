from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import base64
import whisper
import tempfile
import traceback

app = Flask(__name__)

# Set Azure OpenAI credentials for GPT-4o
os.environ["OPENAI_API_TYPE"] = "OPENAI_API_TYPE"
os.environ["AZURE_OPENAI_API_KEY"] = "AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_OPENAI_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "AZURE_OPENAI_API_VERSION"

# 🎯 Set ElevenLabs API key
elevenlabs_client = ElevenLabs(api_key="api_key")

# Initialize clients
try:
    chat_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip('/')
    )
    whisper_model = whisper.load_model("base")
except Exception as e:
    print("Initialization error:", e)
    raise

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_deployments', methods=['GET'])
def test_deployments():
    # Test GPT‑4o
    try:
        chat_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        gpt_status = "✅ GPT‑4o working"
    except Exception as e:
        gpt_status = f"❌ GPT‑4o error: {e}"

    # Test Whisper
    try:
        import wave, struct
        with wave.open("test.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(struct.pack('<h', 0) * 16000)
        whisper_model.transcribe("test.wav")
        whisper_status = "✅ Whisper working"
        os.remove("test.wav")
    except Exception as e:
        whisper_status = f"❌ Whisper error: {e}"

    # Test ElevenLabs
    try:
        audio_stream = elevenlabs_client.text_to_speech.convert(
            text="Hello from ElevenLabs!",
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
            voice_settings=VoiceSettings()
        )
        audio_bytes = b"".join(audio_stream)
        eleven_status = "✅ ElevenLabs working"
    except Exception as e:
        eleven_status = f"❌ ElevenLabs error: {e}"

    return jsonify({
        "gpt_status": gpt_status,
        "whisper_status": whisper_status,
        "elevenlabs_status": eleven_status,
    })

@app.route('/process_voice', methods=['POST'])
def process_voice():
    temp_path = None
    try:
        f = request.files['audio']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
            f.save(tmp.name)
            temp_path = tmp.name

        transcript = whisper_model.transcribe(temp_path)["text"]

        conversation_history.append({"role": "user", "content": transcript})
        if len(conversation_history) > 6:
            conversation_history.pop(0)

        resp = chat_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}] + conversation_history,
            temperature=0.7,
            max_tokens=800
        )
        bot_text = resp.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": bot_text})

        audio_stream = elevenlabs_client.text_to_speech.convert(
            text=bot_text,
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model_id="eleven_multilingual_v2",
            output_format="mp3_22050_32",
            voice_settings=VoiceSettings()
        )
        audio_bytes = b"".join(audio_stream)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            "success": True,
            "transcript": transcript,
            "response": bot_text,
            "audio": audio_base64
        })

    except Exception as e:
        traceback.print_exc()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
