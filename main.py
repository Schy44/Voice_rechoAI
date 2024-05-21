import tensorflow as tf
import numpy as np
import sounddevice as sd
import webbrowser
import os
import pyttsx3
import noisereduce as nr

a = pyttsx3.init('sapi5')
voices = a.getProperty('voices')
a.setProperty('voice', voices[1].id)

def speak(audio):
    a.say(audio)
    a.runAndWait()

# Load the SavedModel
saved_model_path = "E:/2024Uni/AI/saved_model"
loaded_model = tf.saved_model.load(saved_model_path)

commands = np.array(['no', 'yes', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'])

def record_audio(duration=1, sample_rate=16000):
    print("Press Enter to start recording...")
    input()  # Wait for the user to press Enter
    print("Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def get_spectrogram(waveform):
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def predict_command(waveform):
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, 0)
    
    # Use the "serving_default" signature for inference
    infer = loaded_model.signatures["serving_default"]
    prediction = infer(spectrogram)
    
    predicted_label = tf.argmax(prediction['output_0'][0])
    return commands[predicted_label]

def is_silence(audio, threshold=0.01):
    return np.max(np.abs(audio)) < threshold

def preprocess_audio(audio, sample_rate=16000):
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=audio, sr=sample_rate)
    return reduced_noise

def main():
    while True:
        speak("Press Enter to give your command or type 'exit' to quit.")
        user_input = input("Press Enter to start recording or type 'exit' to quit: ").strip().lower()
        
        if user_input == 'exit':
            break
        
        waveform = record_audio()
        if is_silence(waveform):
            print("No significant audio detected, please try again.")
            speak("No significant audio detected, please try again.")
            continue
        
        # Preprocess the audio
        waveform = preprocess_audio(waveform)
        
        print(f"Recorded waveform: {waveform}")

        command = predict_command(waveform)
        print(f"Command is: {command}\n")
        speak(f"Command is: {command}")

        if 'go' in command:
            webbrowser.open("https://www.youtube.com")
            speak("You may start watching videos on YouTube")
        elif 'yes' in command:
            project_directory = r"E:\2024Uni\WebProg\bmicalculator"
            os.chdir(project_directory)
            os.system('npm start')
            speak("React app is running")
        elif 'no' in command:
            project_directory = r"E:\2024Uni\WebProg\bmicalculator"
            os.chdir(project_directory)
            os.system('npm start')
        
            speak("React app is running")    
        elif 'up' in command:
            project_directory = r"E:\2024Uni\WebProg\bmicalculator"
            os.chdir(project_directory)
            os.system('npm start')
            speak("React app is running")    
            

if __name__ == "__main__":
    main()
