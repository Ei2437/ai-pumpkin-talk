# -*- coding: utf-8 -*-
# 最適化版 2t.py - パフォーマンス改善 + 出力整形

import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
import keyboard
from threading import Thread, Lock
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
    MONITOR_INTERVAL: float = 0.3

# ==== グローバル状態 ====
class GlobalState:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.audio_lock = Lock()
        self.stream: Optional[sd.InputStream] = None
        self.monitoring = True
        self.last_response = ""

g_state = GlobalState()
config = ServerConfig()

# HTTPセッション（再利用）
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=2
)
session.mount('http://', adapter)

# ==== 出力整形関数 ====
def print_header():
    """ヘッダー表示"""
    print("\n" + "=" * 70)
    print("AI Pumpkin Controller 起動完了".center(70))
    print("=" * 70)
    print("\n【操作方法】")
    print("  Space      : 音声録音")
    print("  Q          : 緊急スキップ (コンプライアンス用)")
    print("  Left/Right : カメラ切替")
    print("  A/K/L/数字 : その他キー送信")
    print("=" * 70 + "\n")

def print_recording_start():
    """録音開始"""
    print("録音開始...", end="", flush=True)

def print_recording_end():
    """録音終了"""
    print(" 完了")

def print_recognition(text: str):
    """音声認識結果"""
    print(f"\n認識: {text}")

def print_send_complete(text: str):
    """送信完了"""
    print(f"送信完了")

def print_response(text: str):
    """AI応答"""
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
    print()  # 空行

def print_exit():
    """終了メッセージ"""
    print("\n" + "=" * 70)
    print("終了しました".center(70))
    print("=" * 70 + "\n")

def print_error(message: str):
    """エラー表示"""
    print(f"エラー: {message}")

# ==== 通信関数（最適化版） ====
def send_key(key: str):
    """キーイベントを送信"""
    try:
        session.post(
            config.KEY_URL,
            json={"key": key},
            timeout=config.TIMEOUT_KEY
        )
        if key == 'q':
            print("\n緊急スキップ送信")
    except:
        pass  # エラーは無視

def send_text(text: str):
    """テキストを送信"""
    try:
        session.post(
            config.TEXT_URL,
            json={"text": text},
            timeout=config.TIMEOUT_TEXT
        )
        print_send_complete(text)
    except Exception as e:
        print_error(f"送信失敗: {e}")

def monitor_responses():
    """AI応答をポーリングして表示"""
    while g_state.monitoring:
        try:
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
                    
        except:
            pass
        
        time.sleep(config.MONITOR_INTERVAL)

# ==== キー処理 ====
def on_key(event):
    """キー押下時の処理"""
    if event.event_type != keyboard.KEY_DOWN:
        return
    
    key = event.name
    if key in ['left', 'right', 'a', 'k', 'l', 'q', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        Thread(target=send_key, args=(key,), daemon=True).start()

# ==== 音声処理 ====
def audio_callback(indata, frames, time_info, status):
    """音声データのコールバック"""
    if g_state.recording:
        with g_state.audio_lock:
            g_state.audio_frames.append(indata.copy())

def transcribe(audio_data: np.ndarray) -> Optional[str]:
    """音声認識"""
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

# ==== メイン処理 ====
def main():
    print_header()
    
    # AI応答モニタリングスレッド起動
    Thread(target=monitor_responses, daemon=True).start()
    
    # キーフック登録
    keyboard.hook(on_key)
    
    # 音声ストリーム開始
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
            # Spaceキーの状態をチェック
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
                    
                    # 音声処理を別スレッドで実行
                    with g_state.audio_lock:
                        data = np.concatenate(g_state.audio_frames) if g_state.audio_frames else None
                    
                    if data is not None and len(data) > 0:
                        def process():
                            text = transcribe(data)
                            if text:
                                print_recognition(text)
                                send_text(text)
                        
                        Thread(target=process, daemon=True).start()
            
            # CPU負荷を下げる
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