from openai import AsyncOpenAI
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import asyncio
import os
import config
import azure.cognitiveservices.speech as speechsdk

BEEP = AudioSegment.from_file("audio/beep.mp3", format="mp3")
BELL = AudioSegment.from_file("audio/bell.mp3", format="mp3")
ERROR = AudioSegment.from_file("audio/error.mp3", format="mp3")

# OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY", config.OPENAI_API_KEY)
client = AsyncOpenAI(api_key=api_key)

# Define Azure speech config
azure_api_key = os.environ.get(
    "AZURE_API_KEY", config.AZURE_API_KEY
)  # config.azure_api_key
azure_region = os.environ.get(
    "AZURE_REGION", config.AZURE_REGION
)  # config.azure_region
speech_config = speechsdk.SpeechConfig(subscription=azure_api_key, region=azure_region)


def lower_audio(audio, deafen):
    loudness = audio.dBFS
    adjustment_factor = deafen - loudness
    return audio + adjustment_factor


ERROR = lower_audio(ERROR, -30)
BELL = lower_audio(BELL, -25)


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


# Initialize the recognizer
r = sr.Recognizer()


# Function to listen for the wake word
def listen_for_commands(commands):
    with sr.Microphone(device_index=1) as source:
        print("Listening for 'OK Chat'...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
        if "ok chat" in command:
            return True
        elif "stop" in command:
            print("received stop")
            commands["stop"] = True
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
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
    play(audio_segment)


def transcribe_audio(speech_config):
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    result = speech_recognizer.recognize_once_async().get()
    return result.text.strip()


async def listening():
    play(BEEP)
    print("Listening for a prompt...")
    input_text = transcribe_audio(speech_config)
    try:
        command = input_text.lower()
        print(f"You said: {command}")

        # Send the command to ChatGPT for processing
        text_response = await generate_response(command)
        print(f"ChatGPT: {text_response}")

        # Convert the response to speech and play it
        play(BELL)
        await gpt_speech(text_response)

    except sr.UnknownValueError:
        await gpt_speech("Sorry, I didn't catch that. Please repeat.")
        await listening()
    except sr.RequestError:
        await gpt_speech("Sorry, I'm having trouble processing your request.")


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
