import speech_recognition as sr

r = sr.Recognizer()
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")
