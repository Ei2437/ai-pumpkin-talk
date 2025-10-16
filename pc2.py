import os
import time
import json
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io import wavfile
import keyboard

def start_recording():
    print("録音開始...")
    audio_frames = []
    recording_stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.int16)
    recording_stream.start()
    return recording_stream, audio_frames

def stop_recording(recording_stream, audio_frames):
    print("録音終了...")
    recording_stream.stop()
    
    if audio_frames:
        audio_data = np.concatenate(audio_frames, axis=0)
        audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
        recording_stream.close()
        return audio
    
    recording_stream.close()
    return None

def transcribe_audio(audio):
    recognizer = sr.Recognizer()
    try:
        print("文字起こし中...")
        text = recognizer.recognize_google(audio, language="ja-JP")
        print(f"認識されたテキスト: {text}")
        return text
    except sr.UnknownValueError:
        print("音声を認識できませんでした")
        return None
    except sr.RequestError as e:
        print(f"音声認識サービスでエラーが発生しました: {e}")
        return None

def send_text_to_server(text):
    url = "http://localhost:5000/receive_text"  # pc1.py のアドレス
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("サーバーにテキストを送信しました")
    except requests.exceptions.RequestException as e:
        print(f"サーバー送信でエラーが発生しました: {e}")

def main():
    print("Spaceキーを押して録音開始...")
    is_recording = False
    last_key_state = False
    audio_frames = []
    recording_stream = None

    while True:
        current_key_state = keyboard.is_pressed('space')

        if current_key_state != last_key_state:
            if current_key_state:
                if not is_recording:
                    # 録音開始
                    recording_stream, audio_frames = start_recording()
                    is_recording = True
                else:
                    # 録音停止
                    audio = stop_recording(recording_stream, audio_frames)
                    if audio:
                        text = transcribe_audio(audio)
                        if text:
                            send_text_to_server(text)
                    is_recording = False
            last_key_state = current_key_state

        if is_recording:
            data, overflowed = recording_stream.read(1024)
            if not overflowed:
                audio_frames.append(data)

        time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了")