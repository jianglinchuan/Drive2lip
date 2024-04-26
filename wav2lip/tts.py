import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume',0.6)
voices = engine.getProperty('voices') 
for i in voices:
    print(i)
engine.setProperty('voice',voices[6].id) 
engine.save_to_file("Nice to meet you", "voice\\English_male.wav")
engine.runAndWait()
engine.stop()