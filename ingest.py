from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os


# Load env variables
load_dotenv()

# Debug check
print("Groq key loaded:", bool(os.getenv("GROQ_API_KEY")))

# Groq OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Paths
AUDIO_PATH = Path("../data/audio/video.mp3")
TEXT_PATH = Path("../data/text/video.txt")

TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Starting transcription with Groq...")

with open(AUDIO_PATH, "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3"
    )

with open(TEXT_PATH, "w", encoding="utf-8") as f:
    f.write(transcript.text)

print("✅ Transcription saved to data/text/video.txt")
