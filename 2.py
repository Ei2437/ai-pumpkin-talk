# pc2.py - qキー緊急スキップ対応版
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import keyboard
from threading import Thread, Lock
import time

# 設定
SERVER_URL = "http://sudume.hamako-ths.ed.jp:5000/receive_text"
KEY_URL = "http://sudume.hamako-ths.ed.jp:5001/key_event"
MONITOR_URL = "http://sudume.hamako-ths.ed.jp:5002/get_response"  # 新規: モニター用
SAMPLE_RATE = 16000

# グローバル変数
recording = False
audio_frames = []
audio_lock = Lock()
stream = None
monitoring = True

def send_key(key):
    """キーイベントを送信（最小構成）"""
    try:
        requests.post(KEY_URL, json={"key": key}, timeout=0.5)
        if key == 'q':
            print("[緊急スキップ送信]")
    except:
        pass  # エラーは無視してレスポンスを待たない

def send_text(text):
    """テキストを送信"""
    try:
        requests.post(SERVER_URL, json={"text": text}, timeout=2)
        print(f"送信完了: {text}")
    except Exception as e:
        print(f"送信エラー: {e}")

def monitor_responses():
    """AI応答をポーリングして表示"""
    global monitoring
    last_response = ""
    
    while monitoring:
        try:
            response = requests.get(MONITOR_URL, timeout=1)
            if response.status_code == 200:
                data = response.json()
                current_response = data.get("response", "")
                
                if current_response and current_response != last_response:
                    print(f"\n                                                   パンプキン: {current_response}\n")
                    last_response = current_response
                    
        except:
            pass
        
        time.sleep(0.3)  # 300msごとにポーリング

def on_key(event):
    """キー押下時の処理（イベント駆動）"""
    if event.event_type != keyboard.KEY_DOWN:
        return
    
    key = event.name
    # q キーを追加
    if key in ['left', 'right', 'a', 'k', 'l', 'q', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
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
    global recording, audio_frames, stream, monitoring
    
    print("=" * 60)
    print("AI Pumpkin Controller 起動完了")
    print("=" * 60)
    print("操作方法:")
    print("   Space: 音声録音")
    print("   Q: 緊急スキップ (コンプライアンス用)")
    print("   Left/Right/A/K/L/数字: その他キー送信")
    print("=" * 60)
    print()
    
    # AI応答モニタリングスレッド起動
    Thread(target=monitor_responses, daemon=True).start()
    
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
                                print(f"                                                                   認識: {text}")
                                send_text(text)
                        
                        Thread(target=process, daemon=True).start()
            
            # CPU負荷を下げる
            keyboard.read_event(suppress=False)
            
    except KeyboardInterrupt:
        print("\n終了")
        monitoring = False
    finally:
        keyboard.unhook_all()
        if stream:
            stream.stop()
            stream.close()

if __name__ == "__main__":
    main()