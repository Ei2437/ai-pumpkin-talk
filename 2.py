# pc2.py - 最適化版
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import keyboard
from threading import Thread, Lock

# 設定
SERVER_URL = "http://sudume.hamako-ths.ed.jp:5000/receive_text"
KEY_URL = "http://sudume.hamako-ths.ed.jp:5001/key_event"
SAMPLE_RATE = 16000

# グローバル変数
recording = False
audio_frames = []
audio_lock = Lock()
stream = None

def send_key(key):
    """キーイベントを送信（最小構成）"""
    try:
        requests.post(KEY_URL, json={"key": key}, timeout=0.5)
    except:
        pass  # エラーは無視してレスポンスを待たない

def send_text(text):
    """テキストを送信"""
    try:
        requests.post(SERVER_URL, json={"text": text}, timeout=2)
        print(f"送信完了: {text}")
    except Exception as e:
        print(f"送信エラー: {e}")

def on_key(event):
    """キー押下時の処理（イベント駆動）"""
    if event.event_type != keyboard.KEY_DOWN:
        return
    
    key = event.name
    # 対象キーのみ処理
    if key in ['left', 'right', 'a', 'k', 'l', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        # 非同期で送信（ブロックしない）
        Thread(target=send_key, args=(key,), daemon=True).start()

def audio_callback(indata, frames, time, status):
    """音声データのコールバック"""
    if recording:
        with audio_lock:
            audio_frames.append(indata.copy())

def transcribe(audio_data):
    """音声認識"""
    recognizer = sr.Recognizer()
    try:
        audio = sr.AudioData(audio_data.tobytes(), SAMPLE_RATE, 2)
        text = recognizer.recognize_google(audio, language="ja-JP")
        return text
    except:
        return None

def main():
    global recording, audio_frames, stream
    
    print("起動完了")
    print("Space: 録音 | Left/Right/A/K/L/数字: キー送信")
    
    # キーフックを登録（軽量）
    keyboard.hook(on_key)
    
    # 音声ストリーム開始
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        callback=audio_callback
    )
    stream.start()
    
    try:
        while True:
            # Spaceキーの状態をポーリング（軽量）
            if keyboard.is_pressed('space'):
                if not recording:
                    print("録音開始...")
                    with audio_lock:
                        audio_frames = []
                    recording = True
            else:
                if recording:
                    print("録音終了...")
                    recording = False
                    
                    # 音声処理を別スレッドで実行
                    with audio_lock:
                        data = np.concatenate(audio_frames) if audio_frames else None
                    
                    if data is not None and len(data) > 0:
                        def process():
                            text = transcribe(data)
                            if text:
                                print(f"認識: {text}")
                                send_text(text)
                        
                        Thread(target=process, daemon=True).start()
            
            # CPU負荷を下げる
            keyboard.read_event(suppress=False)
            
    except KeyboardInterrupt:
        print("\n終了")
    finally:
        keyboard.unhook_all()
        if stream:
            stream.stop()
            stream.close()

if __name__ == "__main__":
    main()