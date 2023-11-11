from openai import AsyncOpenAI
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import subprocess
import tempfile
import asyncio
import os

BEEP = AudioSegment.from_file("audio/beep.mp3", format="mp3")
BELL = AudioSegment.from_file("audio/bell.mp3", format="mp3")
ERROR = AudioSegment.from_file("audio/error.mp3", format="mp3")
DEAFEN = -30

# Replace with your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)


def lower_audio(audio):
    loudness = audio.dBFS
    adjustment_factor = DEAFEN - loudness
    return audio + adjustment_factor


ERROR = lower_audio(ERROR)


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


# Function to convert text to speech using gTTS
def speak(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        subprocess.run(["mpg123", fp.name])  # Requires 'mpg123' to be installed


# Initialize the recognizer
r = sr.Recognizer()


# Function to listen for the wake word
def listen_for_commands(commands):
    with sr.Microphone(device_index=1) as source:
        print("Listening for 'OK Chat'...")
        r.adjust_for_ambient_noise(source)
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


async def listening():
    play(BEEP)
    print("Listening for a command...")
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
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
