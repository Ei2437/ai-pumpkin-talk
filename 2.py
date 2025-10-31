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
import queue

# ==== 設定 ====
@dataclass(frozen=True)
class ServerConfig:
    TEXT_URL: str = "http://sudume.hamako-ths.ed.jp:5000/receive_text"
    KEY_URL: str = "http://sudume.hamako-ths.ed.jp:5001/key_event"
    MONITOR_URL: str = "http://sudume.hamako-ths.ed.jp:5002/get_response"
    SAMPLE_RATE: int = 16000
    TIMEOUT_KEY: float = 1.0  # キー送信タイムアウト延長
    TIMEOUT_TEXT: float = 3.0  # テキスト送信タイムアウト延長
    TIMEOUT_MONITOR: float = 2.0
    MONITOR_INTERVAL: float = 1.5
    MONITOR_IDLE_INTERVAL: float = 3.0
    MAX_RETRIES: int = 3  # リトライ回数
    RETRY_DELAY: float = 0.5  # リトライ間隔

# ==== グローバル状態 ====
class GlobalState:
    def __init__(self):
        self.recording = False
        self.audio_frames = []
        self.audio_lock = Lock()
        self.stream: Optional[sd.InputStream] = None
        self.monitoring = True
        self.last_response = ""
        self.is_speaking = False
        self.speaking_check_event = Event()
        
        # 送信キュー（確実に送信するため）
        self.key_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # 接続状態
        self.server_connected = False
        self.connection_check_time = 0

g_state = GlobalState()
config = ServerConfig()

# セッション設定（接続プール・リトライ強化）
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=requests.adapters.Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
)
session.mount('http://', adapter)
session.mount('https://', adapter)

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
    print("🎤 録音開始...", end="", flush=True)
def print_recording_end():
    print(" ✓ 完了")
def print_recognition(text: str):
    print(f"\n📝 認識: {text}")
def print_send_complete(text: str):
    print(f"✓ 送信完了")
def print_response(text: str):
    max_width = 70
    lines = []
    current_line = ""
    
    for char in text:
        current_line += char
        if len(current_line) >= max_width and char in ['。', '!', '?', '！', '？', '\n']:
            lines.append(current_line)
            current_line = ""
    
    if current_line:
        lines.append(current_line)
    
    print(f"\n🎃 パンプキン:")
    for line in lines:
        print(f"   {line}")
    print()

def print_exit():
    print("\n" + "=" * 70)
    print("終了しました".center(70))
    print("=" * 70 + "\n")
def print_error(message: str):
    print(f"❌ エラー: {message}")
def print_warning(message: str):
    print(f"⚠️  警告: {message}")
def print_info(message: str):
    print(f"ℹ️  情報: {message}")

# ==== サーバー接続確認 ====
def check_server_connection() -> bool:
    """サーバーとの接続を確認"""
    try:
        response = session.get(
            config.MONITOR_URL,
            timeout=2.0
        )
        return response.status_code == 200
    except:
        return False

def connection_monitor():
    """接続状態を定期的に監視"""
    while g_state.monitoring:
        current_time = time.time()
        if current_time - g_state.connection_check_time > 10.0:
            connected = check_server_connection()
            if connected != g_state.server_connected:
                g_state.server_connected = connected
                if connected:
                    print_info("サーバーに接続しました")
                else:
                    print_warning("サーバーとの接続が切れています")
            g_state.connection_check_time = current_time
        time.sleep(5.0)

# ==== 通信系（リトライ機能付き） ====
def send_key_with_retry(key: str) -> bool:
    """キー送信（リトライ付き）"""
    for attempt in range(config.MAX_RETRIES):
        try:
            response = session.post(
                config.KEY_URL,
                json={"key": key},
                timeout=config.TIMEOUT_KEY
            )
            if response.status_code == 200:
                if key == 'q':
                    print("\n🚨 緊急スキップ送信")
                elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                    print(f"🔢 数字キー送信: {key}")
                return True
            else:
                print_warning(f"キー送信失敗 (ステータス: {response.status_code})")
        except requests.exceptions.Timeout:
            print_warning(f"キー送信タイムアウト (試行 {attempt + 1}/{config.MAX_RETRIES})")
        except Exception as e:
            print_error(f"キー送信エラー: {e}")
        
        if attempt < config.MAX_RETRIES - 1:
            time.sleep(config.RETRY_DELAY)
    
    print_error(f"キー送信失敗: {key}")
    return False

def send_text_with_retry(text: str) -> bool:
    """テキスト送信（リトライ付き）"""
    for attempt in range(config.MAX_RETRIES):
        try:
            response = session.post(
                config.TEXT_URL,
                json={"text": text},
                timeout=config.TIMEOUT_TEXT
            )
            if response.status_code == 200:
                print_send_complete(text)
                g_state.is_speaking = True
                g_state.speaking_check_event.set()
                return True
            else:
                print_warning(f"テキスト送信失敗 (ステータス: {response.status_code})")
        except requests.exceptions.Timeout:
            print_warning(f"テキスト送信タイムアウト (試行 {attempt + 1}/{config.MAX_RETRIES})")
        except Exception as e:
            print_error(f"テキスト送信エラー: {e}")
        
        if attempt < config.MAX_RETRIES - 1:
            time.sleep(config.RETRY_DELAY)
    
    print_error(f"テキスト送信失敗: {text}")
    return False

def key_sender_worker():
    """キュー内のキーを順次送信"""
    while g_state.monitoring:
        try:
            key = g_state.key_queue.get(timeout=0.1)
            send_key_with_retry(key)
        except queue.Empty:
            continue

def text_sender_worker():
    """キュー内のテキストを順次送信"""
    while g_state.monitoring:
        try:
            text = g_state.text_queue.get(timeout=0.1)
            send_text_with_retry(text)
        except queue.Empty:
            continue

def monitor_responses():
    """応答モニタリング"""
    consecutive_same_count = 0
    
    while g_state.monitoring:
        try:
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
                    g_state.is_speaking = False
                elif current_response == g_state.last_response:
                    consecutive_same_count += 1
                    if consecutive_same_count >= 5:
                        g_state.is_speaking = False
                    
        except:
            pass
        
        time.sleep(wait_time)

# ==== キー処理 ====
def on_key(event):
    """キーボードイベント処理"""
    if event.event_type != keyboard.KEY_DOWN:
        return
    
    key = event.name
    if key in ['left', 'right', 'a', 'k', 'l', 'q', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        # キューに追加（確実に送信するため）
        g_state.key_queue.put(key)

# ==== 音声処理 ====
def audio_callback(indata, frames, time_info, status):
    """音声入力コールバック"""
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

# ==== main ====
def main():
    print_header()
    
    # サーバー接続確認
    print_info("サーバーへの接続を確認中...")
    if not check_server_connection():
        print_warning("サーバーに接続できません。続行しますか？ (y/n)")
        response = input().lower()
        if response != 'y':
            print_exit()
            return
    else:
        g_state.server_connected = True
        print_info("サーバー接続OK")
    
    # 各種ワーカースレッド起動
    Thread(target=connection_monitor, daemon=True).start()
    Thread(target=monitor_responses, daemon=True).start()
    Thread(target=key_sender_worker, daemon=True).start()
    Thread(target=text_sender_worker, daemon=True).start()
    
    # キーボードフック
    keyboard.hook(on_key)
    
    # 音声デバイス初期化
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
    
    print_info("準備完了！操作を開始してください")
    
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
                                # キューに追加（確実に送信）
                                g_state.text_queue.put(text)
                        
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