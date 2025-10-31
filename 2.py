# -*- coding: utf-8 -*-
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import keyboard
from threading import Thread, Lock, Event
import time
import sys
from typing import Optional
from dataclasses import dataclass

# ==== 設定 ====
@dataclass(frozen=True)
class ServerConfig:
    TEXT_URL: str = "http://sudume.hamako-ths.ed.jp:5000/receive_text"
    KEY_URL: str = "http://sudume.hamako-ths.ed.jp:5001/key_event"
    MONITOR_URL: str = "http://sudume.hamako-ths.ed.jp:5002/get_response"
    SAMPLE_RATE: int = 16000
    TIMEOUT_KEY: float = 0.3
    TIMEOUT_TEXT: float = 2.0
    TIMEOUT_MONITOR: float = 1.0
    MONITOR_INTERVAL: float = 1.5  # 3回/秒 → 0.67回/秒に変更
    MONITOR_IDLE_INTERVAL: float = 3.0  # アイドル時はさらに間隔を広げる

# ==== グローバル状態 ====
class GlobalState:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.audio_lock = Lock()
        self.stream: Optional[sd.InputStream] = None
        self.monitoring = True
        self.last_response = ""
        self.is_speaking = False  # サーバーが話している状態
        self.speaking_check_event = Event()  # 話し中検知用

g_state = GlobalState()
config = ServerConfig()

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=2
)
session.mount('http://', adapter)

# ==== ヘッダー ====
def print_header():
    print("\n" + "=" * 70)
    print("AI Pumpkin Controller 起動完了".center(70))
    print("=" * 70)
    print("\n【操作方法】")
    print("  Space      : 音声録音")
    print("  Q          : 緊急スキップ")
    print("  Left/Right : モーション")
    print("  A/K/L/数字 : その他キー送信")
    print("=" * 70 + "\n")

def print_recording_start():
    print("録音開始...", end="", flush=True)
def print_recording_end():
    print(" 完了")
def print_recognition(text: str):
    print(f"\n認識: {text}")
def print_send_complete(text: str):
    print(f"送信完了")
def print_response(text: str):
    # 長い応答の場合は折り返し
    max_width = 70
    lines = []
    current_line = ""
    
    for char in text:
        current_line += char
        if len(current_line) >= max_width and char in ['。', '！', '？', '!', '?', '\n']:
            lines.append(current_line)
            current_line = ""
    
    if current_line:
        lines.append(current_line)
    
    print(f"\nパンプキン:")
    for line in lines:
        print(f"   {line}")
    print()

def print_exit():
    print("\n" + "=" * 70)
    print("終了しました".center(70))
    print("=" * 70 + "\n")
def print_error(message: str):
    print(f"エラー: {message}")

# ==== 通信系 ====
def send_key(key: str):
    try:
        session.post(
            config.KEY_URL,
            json={"key": key},
            timeout=config.TIMEOUT_KEY
        )
        if key == 'q':
            print("\n緊急スキップ送信")
    except:
        pass

def send_text(text: str):
    try:
        session.post(
            config.TEXT_URL,
            json={"text": text},
            timeout=config.TIMEOUT_TEXT
        )
        print_send_complete(text)
        # テキスト送信したら話し中フラグを立てる
        g_state.is_speaking = True
        g_state.speaking_check_event.set()
    except Exception as e:
        print_error(f"送信失敗: {e}")

def monitor_responses():
    consecutive_same_count = 0
    
    while g_state.monitoring:
        try:
            # 話し中は短い間隔、アイドル時は長い間隔
            if g_state.is_speaking:
                wait_time = config.MONITOR_INTERVAL
            else:
                wait_time = config.MONITOR_IDLE_INTERVAL
            
            response = session.get(
                config.MONITOR_URL,
                timeout=config.TIMEOUT_MONITOR
            )
            if response.status_code == 200:
                data = response.json()
                current_response = data.get("response", "")
                
                if current_response and current_response != g_state.last_response:
                    print_response(current_response)
                    g_state.last_response = current_response
                    consecutive_same_count = 0
                    g_state.is_speaking = False  # 応答取得完了
                elif current_response == g_state.last_response:
                    consecutive_same_count += 1
                    # 同じ応答が5回続いたらアイドル状態と判断
                    if consecutive_same_count >= 5:
                        g_state.is_speaking = False
                    
        except:
            pass
        
        time.sleep(wait_time)

# ==== キー処理 ====
def on_key(event):
    if event.event_type != keyboard.KEY_DOWN:
        return
    
    key = event.name
    if key in ['left', 'right', 'a', 'k', 'l', 'q', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        Thread(target=send_key, args=(key,), daemon=True).start()

# ==== 音声処理 ====
def audio_callback(indata, frames, time_info, status):
    if g_state.recording:
        with g_state.audio_lock:
            g_state.audio_frames.append(indata.copy())

def transcribe(audio_data: np.ndarray) -> Optional[str]:
    recognizer = sr.Recognizer()
    try:
        audio = sr.AudioData(audio_data.tobytes(), config.SAMPLE_RATE, 2)
        text = recognizer.recognize_google(audio, language="ja-JP")
        return text
    except sr.UnknownValueError:
        print_error("音声を認識できませんでした")
        return None
    except sr.RequestError as e:
        print_error(f"音声認識サービスエラー: {e}")
        return None
    except Exception as e:
        print_error(f"音声認識エラー: {e}")
        return None

# ==== main ====
def main():
    print_header()
    Thread(target=monitor_responses, daemon=True).start()
    keyboard.hook(on_key)
    try:
        g_state.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            callback=audio_callback
        )
        g_state.stream.start()
    except Exception as e:
        print_error(f"音声デバイスの初期化に失敗: {e}")
        return
    
    try:
        while True:
            if keyboard.is_pressed('space'):
                if not g_state.recording:
                    print_recording_start()
                    with g_state.audio_lock:
                        g_state.audio_frames = []
                    g_state.recording = True
            else:
                if g_state.recording:
                    print_recording_end()
                    g_state.recording = False
                    
                    with g_state.audio_lock:
                        data = np.concatenate(g_state.audio_frames) if g_state.audio_frames else None
                    
                    if data is not None and len(data) > 0:
                        def process():
                            text = transcribe(data)
                            if text:
                                print_recognition(text)
                                send_text(text)
                        
                        Thread(target=process, daemon=True).start()
            
            keyboard.read_event(suppress=False)
            
    except KeyboardInterrupt:
        print_exit()
        g_state.monitoring = False
    finally:
        keyboard.unhook_all()
        if g_state.stream:
            g_state.stream.stop()
            g_state.stream.close()

if __name__ == "__main__":
    main()