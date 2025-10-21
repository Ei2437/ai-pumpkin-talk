# pc2.py
import os
import time
import json
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io import wavfile
import keyboard # keyboard ライブラリ追加

# Aキーの押下状態を追跡
a_key_pressed = False

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
    url = "http://sudume.hamako-ths.ed.jp:5000/receive_text"  # pc1.py のアドレス
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("サーバーで処理が完了しました。")
    except requests.exceptions.ConnectionError:
        print("サーバーに接続できません。pc1.py が起動しているか確認してください。")
    except requests.exceptions.Timeout:
        print("サーバーへのリクエストがタイムアウトしました。")
    except requests.exceptions.RequestException as e:
        print(f"サーバー送信でエラーが発生しました: {e}")

def send_key_to_temp(key, action="down"):
    """temp.pyのFlaskサーバーにキー入力とそのアクションを送信"""
    url = "http://sudume.hamako-ths.ed.jp:5001/key_event" # temp.py のアドレス
    payload = {"key": key, "action": action}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Key '{key}' action '{action}' sent successfully to temp.py server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send key '{key}' action '{action}' to temp.py server: {e}")

def on_key_press(event):
    """keyboard ライブラリのキー押下イベントリスナー"""
    global a_key_pressed
    if event.name == 'space':
        pass # space キーは録音用
    elif event.name == 'left':
        send_key_to_temp('left', 'down')
    elif event.name == 'right':
        send_key_to_temp('right', 'down')
    elif event.name == 'a':
        if not a_key_pressed: # 重複送信防止
            a_key_pressed = True
            send_key_to_temp('a', 'down')

def on_key_release(event):
    """keyboard ライブラリのキー離しイベントリスナー"""
    global a_key_pressed
    if event.name == 'space':
        pass # space キーは録音用
    elif event.name == 'left':
        # LEFTキーの離しは通常不要 (状態遷移で決まる)
        pass
    elif event.name == 'right':
        # RIGHTキーの離しは通常不要 (状態遷移で決まる)
        pass
    elif event.name == 'a':
        if a_key_pressed: # 重複送信防止
            a_key_pressed = False
            send_key_to_temp('a', 'up')

def main():
    print("Spaceキーを押して録音開始...")
    print("Left/Right/Aキーも検知します。")
    is_recording = False
    audio_frames = []
    recording_stream = None

    # キーイベントリスナーを登録 (押下と離し)
    keyboard.on_press(on_key_press)
    keyboard.on_release(on_key_release)

    try:
        while True:
            current_key_state = keyboard.is_pressed('space')

            # spaceキーの押下/離し処理
            if current_key_state and not is_recording:
                # 録音開始
                recording_stream, audio_frames = start_recording()
                is_recording = True
            elif not current_key_state and is_recording:
                # 録音停止
                audio = stop_recording(recording_stream, audio_frames)
                if audio:
                    text = transcribe_audio(audio)
                    if text:
                        send_text_to_server(text)
                is_recording = False
                audio_frames = [] # フレームをクリア
                recording_stream = None

            if is_recording:
                data, overflowed = recording_stream.read(1024)
                if not overflowed:
                    audio_frames.append(data)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n終了")
        if is_recording:
            stop_recording(recording_stream, audio_frames)
    finally:
        # リスナーを解除
        keyboard.unhook_all()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了")