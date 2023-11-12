from openai import AsyncOpenAI
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import asyncio
import pvporcupine
import pyaudio
import os
import config
import azure.cognitiveservices.speech as speechsdk
import numpy as np

# audio constants
SAMPLE_RATE = 16000
FRAME_LENGTH = (
    512  # This value might need to be adjusted based on Porcupine's requirements
)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# porcupine keywords
PORCUPINE_KEYWORDS = [
    "keywords/Exit_en_windows_v3_0_0.ppn",
    "keywords/Ok-Chat_en_windows_v3_0_0.ppn",
]

BEEP = AudioSegment.from_file("audio/beep.mp3", format="mp3")
BELL = AudioSegment.from_file("audio/bell.mp3", format="mp3")
ERROR = AudioSegment.from_file("audio/error.mp3", format="mp3")

# OpenAI API Key
api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
client = AsyncOpenAI(api_key=api_key)

# Define Azure speech config
azure_api_key = os.environ.get("AZURE_API_KEY", config.AZURE_API_KEY)
azure_region = os.environ.get("AZURE_REGION", config.AZURE_REGION)
speech_config = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)

# Picovoice API Key
picovoice_key = os.environ.get("PICOVOICE_API_KEY", config.PICOVOICE_API_KEY)


def lower_audio(audio, deafen):
    loudness = audio.dBFS
    adjustment_factor = deafen - loudness
    return audio + adjustment_factor


ERROR = lower_audio(ERROR, -30)
BELL = lower_audio(BELL, -30)


# Asynchronous function to generate a response from ChatGPT
async def generate_response(prompt):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",  # Adjust the model as needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# Initialize Porcupine
porcupine = pvporcupine.create(
    access_key=picovoice_key, keyword_paths=PORCUPINE_KEYWORDS
)

# Initialize PyAudio and open an audio stream
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=FRAME_LENGTH,
)


def get_next_audio_frame():
    """Capture and return the next audio frame."""
    frame = audio_stream.read(FRAME_LENGTH)
    return np.frombuffer(frame, dtype=np.int16)


def listen_for_commands(commands):
    print("Listening for commands...")
    while True:
        audio_frame = get_next_audio_frame()
        keyword_index = porcupine.process(audio_frame)

        if keyword_index == 1:  # "Ok-Chat"
            print("Ok-Chat detected, proceeding to listen.")
            return True
        elif keyword_index == 0:  # "Exit"
            print("Exit command detected, stopping.")
            commands["stop"] = True
            return False


# Function to play the text response
async def gpt_speech(text):
    audio_response = await client.audio.speech.create(
        model="tts-1", voice="nova", input=text
    )

    # Save the binary content to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(audio_response.content)
        temp_file_path = temp_file.name

    # Load the audio from the saved file and play it
    audio_segment = AudioSegment.from_mp3(temp_file_path)

    # Add a short period of silence at the beginning
    silence = AudioSegment.silent(duration=500)  # 500 ms of silence
    audio_segment = silence + audio_segment

    play(audio_segment)


def transcribe_audio(speech_config):
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = speech_recognizer.recognize_once_async().get()
    return result.text.strip()


async def listening():
    try:
        play(BEEP)
        print("Listening for a prompt...")

        # Apply the timeout only to the user's speech input part
        input_text = await asyncio.wait_for(
            asyncio.to_thread(transcribe_audio, speech_config), timeout=5.0
        )

        print(f"You said: {input_text}")

        # The server response part is outside the timeout block
        text_response = await generate_response(input_text)
        print(f"ChatGPT: {text_response}")

        play(BELL)
        await gpt_speech(text_response)

    except asyncio.TimeoutError:
        await gpt_speech("Listening timed out. Please try again.")
        # Optionally, call listening() again or handle the timeout situation
    except Exception as e:
        await gpt_speech(f"An error occurred: {e}")


# Main loop
async def main():
    commands = {"stop": False}
    while True:
        if listen_for_commands(commands):
            await listening()

        # Exit the program when "stop" is said
        if commands["stop"]:
            break

        await asyncio.sleep(0.1)  # To prevent high CPU usage


asyncio.run(main())
audio_stream.close()
pa.terminate()
