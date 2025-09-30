import os
import nest_asyncio
nest_asyncio.apply()

import asyncio
from io import BytesIO
import struct

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from elevenlabs.client import ElevenLabs
from google import genai
from dotenv import load_dotenv

# ---------- Load environment variables ----------
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLAB_API_KEY = os.getenv("ELEVENLAB_API_KEY")
VOICE_ID = "Mc435dX3Wed3ISvcov0t"  # ElevenLabs target voice

# ---------- Initialize clients ----------
client = genai.Client(api_key=GEMINI_API_KEY)
elevenlabs = ElevenLabs(api_key=ELEVENLAB_API_KEY)

# ---------- Helper functions ----------
def convert_to_wav(audio_data: bytes, sample_rate: int = 24000, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM audio bytes in WAV header."""
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data

# ---------- Telegram Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Salam! Text yazın səsə çevirmək üçün, ya da səs mesajı göndərin onu klonlamaq üçün."
    )

# --- Text to speech handler ---
async def text_to_speech(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    from google.genai import types

    # Step 1: Gemini TTS
    model = "gemini-2.5-pro-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=f"""
        Generate a male Azerbaijani voice with a neutral accent, medium pitch,
        steady rhythm, and moderate natural pace.
        Don't change anything. Read as you see:: {text}
    """)],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Sadaltager")
            )
        ),
    )

    audio_buffer = None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        if chunk.candidates and chunk.candidates[0].content.parts[0].inline_data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            audio_buffer = convert_to_wav(inline_data.data)

    if not audio_buffer:
        await update.message.reply_text("Səs generasiyasında xəta baş verdi.")
        return

    # Step 2: ElevenLabs voice conversion in memory
    audio_stream = elevenlabs.speech_to_speech.convert(
        voice_id=VOICE_ID,
        audio=BytesIO(audio_buffer),  # wrap in BytesIO
        model_id="eleven_multilingual_sts_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_stream)
    bio = BytesIO(audio_bytes)
    bio.name = "voice.mp3"
    bio.seek(0)

    await update.message.reply_audio(bio)

# --- Voice cloning handler ---
async def clone_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.voice:
        await update.message.reply_text("Zəhmət olmasa səs mesajı göndərin.")
        return

    # Download Telegram voice file
    file = await update.message.voice.get_file()
    voice_bytes = await file.download_as_bytearray()

    # Wrap bytes in BytesIO
    voice_stream = BytesIO(voice_bytes)
    voice_stream.name = "voice.ogg"  # Telegram voice is OGG

    # Convert to ElevenLabs voice
    audio_stream = elevenlabs.speech_to_speech.convert(
        voice_id=VOICE_ID,
        audio=voice_stream,
        model_id="eleven_multilingual_sts_v2",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(audio_stream)
    bio = BytesIO(audio_bytes)
    bio.name = "cloned_voice.mp3"
    bio.seek(0)

    await update.message.reply_audio(bio)

# ---------- Build Telegram bot ----------
app_bot = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app_bot.add_handler(CommandHandler("start", start))
app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_to_speech))
app_bot.add_handler(MessageHandler(filters.VOICE, clone_voice))

# ---------- Run bot in background ----------
asyncio.create_task(app_bot.run_polling())

# Optional: minimal FastAPI server for Render/Railway
# import uvicorn
# from fastapi import FastAPI
# app = FastAPI()
# @app.get("/")
# def root(): return {"status": "Bot is running"}
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
