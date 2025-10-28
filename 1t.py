# -*- coding: utf-8 -*-
# 最適化版 1t.py - パフォーマンス改善 + コード整理

import os
import time
import json
import requests
import numpy as np
from scipy.io import wavfile
from flask import Flask, request, jsonify
import subprocess
import pygame
from pygame.locals import *
import sys
import av
import cv2
import math
import threading
import wave
import re
import tempfile
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# ==== 定数定義 ====
@dataclass(frozen=True)
class DisplayConfig:
    WIDTH: int = 1920
    HEIGHT: int = 1020
    BG_VIDEO_PATH: str = "videos/BG.mp4"
    BACK_SPEED_SKIP: int = 10

@dataclass(frozen=True)
class VideoConfig:
    MAIN: str = "videos/Pumpkin-Center.mov"
    FULL2: str = "videos/Pumpkin-Center2Left.mov"
    FULL3: str = "videos/Pumpkin-Left.mov"
    FULL4: str = "videos/Pumpkin-Left2Center.mov"
    FULL5: str = "videos/Pumpkin-Center2Right.mov"
    FULL6: str = "videos/Pumpkin-Right.mov"
    FULL7: str = "videos/Pumpkin-Right2Center.mov"
    ENTRY: str = "videos/Pumpkin-Entry.mov"
    FINISH: str = "videos/Pumpkin-Finish.mov"
    TRANSITION_SPEED: float = 1.0

@dataclass(frozen=True)
class SubtitleConfig:
    # パンプキン応答字幕（下部）
    FONT_SIZE: int = 48
    COLOR: Tuple[int, int, int] = (255, 255, 255)
    BG_COLOR: Tuple[int, int, int, int] = (0, 0, 0, 180)
    MAX_WIDTH: int = 1720
    CHAR_DURATION: float = 0.13
    PUNCTUATION_DURATION: float = 0.43
    
    # ユーザー質問字幕（上部）
    USER_FONT_SIZE: int = 36
    USER_COLOR: Tuple[int, int, int] = (255, 255, 255)
    USER_BG_COLOR: Tuple[int, int, int, int] = (40, 40, 40, 220)
    USER_MAX_WIDTH: int = 1520
    USER_DISPLAY_TIME: float = 4.0
    USER_SLIDE_DURATION: float = 0.3
    USER_FADE_DURATION: float = 0.4

SOUND_FILES = {str(i): f"sounds/sound{i if i > 0 else '0'}.wav" for i in range(10)}
SOUND_FILES['1'] = "sounds/OP1.wav"

# ==== State管理 ====
class State(Enum):
    IDLE = "idle"
    ENTRY = "entry"
    NORMAL = "normal"
    FINISH = "finish"
    FULL2 = "full2"
    FULL3 = "full3"
    FULL4 = "full4"
    FULL5 = "full5"
    FULL6 = "full6"
    FULL7 = "full7"

# ==== グローバル状態（最小限） ====
class GlobalState:
    def __init__(self):
        self.a_key_active = False
        self.state = State.IDLE
        self.is_speaking = False
        self.skip_flag = False
        
        self.current_subtitle = ""
        self.user_subtitle_text = ""
        self.user_subtitle_start_time = 0.0
        self.user_subtitle_active = False
        self.latest_response = ""
        
        # ロック
        self.audio_lock = threading.Lock()
        self.subtitle_lock = threading.Lock()
        self.user_subtitle_lock = threading.Lock()
        self.speaking_lock = threading.Lock()
        self.skip_lock = threading.Lock()
        self.response_lock = threading.Lock()

g_state = GlobalState()

# HTTPセッション（再利用）
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3
)
session.mount('http://', adapter)
session.mount('https://', adapter)

# ==== Config Loader ====
class ConfigLoader:
    def __init__(self, config_path: str = "pumpkin.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def get_character_prompt(self) -> str:
        char = self.config["character"]
        knowledge_dict = self.config.get("knowledge", {})
        knowledge_str = "\n".join(
            f"\n【{category}】\n" + "\n".join(items)
            for category, items in knowledge_dict.items()
        )
        return char["prompt"].format(knowledge=knowledge_str)
    
    def get_ollama_config(self) -> dict:
        return self.config["api"]["ollama"]
    
    def get_voicevox_config(self) -> dict:
        return self.config["api"]["voicevox"]
    
    def get_system_config(self) -> dict:
        return self.config.get("system", {})
    
    def get_advanced_config(self) -> dict:
        return self.config.get("advanced", {})

# ==== PumpkinTalk（最適化版） ====
class PumpkinTalk:
    # 文分割用正規表現（コンパイル済み）
    SENTENCE_SPLITTER = re.compile(r'([。！？!?])')
    
    def __init__(self, config_path: str = "pumpkin.json"):
        self.config_loader = ConfigLoader(config_path)
        self.ollama_config = self.config_loader.get_ollama_config()
        self.voicevox_config = self.config_loader.get_voicevox_config()
        self.system_config = self.config_loader.get_system_config()
        self.advanced_config = self.config_loader.get_advanced_config()
        
        self.ollama_url = self.ollama_config["url"]
        self.voicevox_url = self.voicevox_config["url"]
        self.speaker_id = self.voicevox_config["speaker_id"]
        self.model = self.ollama_config["model"]
        
        self.character_prompt = self.config_loader.get_character_prompt()
        self.conversation_history = []
        
        self.voicevox_settings = self.system_config.get("voicevox", {})
        
        # 字幕管理
        self.subtitle_queue = []
        self.subtitle_queue_lock = threading.Lock()
        self.subtitle_thread: Optional[threading.Thread] = None

    def split_sentences(self, text: str) -> List[str]:
        """文分割（最適化版）"""
        parts = self.SENTENCE_SPLITTER.split(text)
        sentences = []
        temp = ""
        
        for part in parts:
            temp += part
            if part in ['。', '！', '？', '!', '?']:
                if temp.strip():
                    sentences.append(temp.strip())
                    temp = ""
        
        if temp.strip():
            sentences.append(temp.strip())
        
        return sentences

    def generate_response_streaming(self, input_text: str):
        """ストリーミング応答生成"""
        if not input_text:
            yield "何か言ったか?もう一度言ってみろよ!"
            return
        
        try:
            self.conversation_history.append(f"ユーザー: {input_text}")
            recent_history = "\n".join(self.conversation_history[-3:])
            
            payload = {
                "model": self.model,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": True,
                "options": self.ollama_config.get("params", {})
            }
            
            response = session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            buffer = ""
            full_response = ""
            
            for line in response.iter_lines():
                with g_state.skip_lock:
                    if g_state.skip_flag:
                        break
                
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            token = chunk["response"]
                            buffer += token
                            full_response += token
                            
                            if token in ['。', '！', '？', '!', '?', '\n']:
                                if buffer.strip():
                                    sentence = self.filter_response(buffer.strip())
                                    if sentence:
                                        yield sentence
                                    buffer = ""
                        
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            if buffer.strip():
                sentence = self.filter_response(buffer.strip())
                if sentence:
                    yield sentence
            
            self.conversation_history.append(f"パンプキン: {full_response}")
            
            with g_state.response_lock:
                g_state.latest_response = full_response
            
        except Exception as e:
            print(f"[ERROR] Ollama API: {e}")
            yield "ちっ、調子が悪いぞ!もう一度話しかけてみろよ!"

    def text_to_speech_fast(self, text: str) -> Optional[str]:
        """高速音声合成（最適化版）"""
        try:
            # クエリ生成
            query_response = session.post(
                f"{self.voicevox_url}/audio_query",
                params={"text": text, "speaker": self.speaker_id},
                timeout=5
            )
            query_response.raise_for_status()
            query_data = query_response.json()
            
            # 設定適用
            if self.voicevox_settings:
                query_data.update({
                    "speedScale": self.voicevox_settings.get("speed", 1.3),
                    "pitchScale": self.voicevox_settings.get("pitch", 0.0),
                    "intonationScale": self.voicevox_settings.get("intonation", 1.0),
                    "volumeScale": self.voicevox_settings.get("volume", 1.0),
                    "postPhonemeLength": self.voicevox_settings.get("post_phoneme_length", 0.2)
                })
            
            # 音声合成
            synthesis_response = session.post(
                f"{self.voicevox_url}/synthesis",
                params={"speaker": self.speaker_id},
                json=query_data,
                timeout=10
            )
            synthesis_response.raise_for_status()
            
            # 一時ファイル作成
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(synthesis_response.content)
                return tmp.name
            
        except Exception as e:
            print(f"[ERROR] VOICEVOX API: {e}")
            return None

    def get_audio_duration(self, wav_file: str) -> float:
        """音声ファイルの長さを取得"""
        try:
            with wave.open(wav_file, 'rb') as wf:
                return wf.getnframes() / float(wf.getframerate())
        except:
            return 0.0

    def subtitle_worker(self):
        """字幕表示ワーカースレッド"""
        while True:
            with self.subtitle_queue_lock:
                if not self.subtitle_queue:
                    break
                text, duration = self.subtitle_queue.pop(0)
            
            sentences = self.split_sentences(text)
            if not sentences:
                sentences = [text]
            
            for sentence in sentences:
                with g_state.skip_lock:
                    if g_state.skip_flag:
                        with g_state.subtitle_lock:
                            g_state.current_subtitle = ""
                        return
                
                # 表示時間計算
                char_count = len(sentence)
                punctuation_count = sum(sentence.count(c) for c in ['、', '。', '!', '?', '...'])
                sentence_duration = (
                    char_count * SubtitleConfig.CHAR_DURATION +
                    punctuation_count * SubtitleConfig.PUNCTUATION_DURATION
                )
                
                with g_state.subtitle_lock:
                    g_state.current_subtitle = sentence
                
                time.sleep(sentence_duration)
                
                with g_state.subtitle_lock:
                    g_state.current_subtitle = ""

    def display_subtitle_gradually(self, text: str, duration: float):
        """字幕をキューに追加"""
        with self.subtitle_queue_lock:
            self.subtitle_queue.append((text, duration))
            
            if self.subtitle_thread is None or not self.subtitle_thread.is_alive():
                self.subtitle_thread = threading.Thread(
                    target=self.subtitle_worker,
                    daemon=True
                )
                self.subtitle_thread.start()

    def play_audio_with_aplay(self, wav_file: str, show_subtitle: bool = False,
                             subtitle_text: str = "", is_final: bool = False):
        """音声再生（最適化版）"""
        if not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
            return

        try:
            with g_state.speaking_lock:
                if not g_state.is_speaking:
                    g_state.is_speaking = True
                    with g_state.audio_lock:
                        g_state.a_key_active = True
            
            duration = self.get_audio_duration(wav_file)
            
            # 字幕表示
            if show_subtitle and subtitle_text:
                self.display_subtitle_gradually(subtitle_text, duration)
            
            # スキップチェック
            with g_state.skip_lock:
                if g_state.skip_flag:
                    os.unlink(wav_file)
                    return
            
            # 音声再生
            subprocess.run(
                ["aplay", "-q", wav_file],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            os.unlink(wav_file)
                
        except Exception as e:
            print(f"[ERROR] 音声再生: {e}")
        finally:
            if is_final:
                with g_state.speaking_lock:
                    g_state.is_speaking = False
                with g_state.audio_lock:
                    g_state.a_key_active = False

    def process_input_text(self, input_text: str):
        """入力テキスト処理（最適化版）"""
        print(f"[受信] {input_text}")
        
        # スキップフラグリセット
        with g_state.skip_lock:
            g_state.skip_flag = False
        
        # ユーザー字幕表示
        show_user_subtitle(input_text)
        
        # 応答生成
        sentences = list(self.generate_response_streaming(input_text))
        
        # スキップチェック
        with g_state.skip_lock:
            if g_state.skip_flag:
                print("[スキップ] 会話を中断しました")
                with self.subtitle_queue_lock:
                    self.subtitle_queue.clear()
                with g_state.subtitle_lock:
                    g_state.current_subtitle = ""
                with g_state.speaking_lock:
                    g_state.is_speaking = False
                with g_state.audio_lock:
                    g_state.a_key_active = False
                g_state.skip_flag = False
                return
        
        # 音声合成・再生
        total = len(sentences)
        for idx, sentence in enumerate(sentences):
            with g_state.skip_lock:
                if g_state.skip_flag:
                    break
            
            if sentence:
                wav_file = self.text_to_speech_fast(sentence)
                if wav_file:
                    is_final = (idx == total - 1)
                    self.play_audio_with_aplay(
                        wav_file,
                        show_subtitle=True,
                        subtitle_text=sentence,
                        is_final=is_final
                    )

    def filter_response(self, response_text: str) -> str:
        """応答フィルタリング"""
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            for old, new in filtering.get("replace_patterns", {}).items():
                response_text = response_text.replace(old, new)
        
        return response_text.strip()

# ==== ユーザー字幕表示 ====
def show_user_subtitle(text: str):
    """ユーザーの質問を上部に表示"""
    with g_state.user_subtitle_lock:
        g_state.user_subtitle_text = text
        g_state.user_subtitle_start_time = time.time()
        g_state.user_subtitle_active = True

# ==== 浮遊モーション（最適化版） ====
def float_motion(t: int, seed: int = 0, amp_y: int = 16,
                amp_x: int = 7, base_speed: float = 0.0007) -> Tuple[int, int]:
    """浮遊モーション計算"""
    dy = int(math.sin(t * base_speed + seed) * amp_y)
    dx = int(
        math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x +
        math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6 +
        math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
    )
    return dx, dy

# ==== PyAV動画クラス（最適化版） ====
class AlphaVideo:
    def __init__(self, path: str):
        container = av.open(path)
        stream = container.streams.video[0]
        self.frames = [frame.to_ndarray(format="rgba") for frame in container.decode(stream)]
        self.total = len(self.frames)
        self.current_frame = 0
        self.direction = 1
        self.frame_accumulator = 0.0
        container.close()

    def get_frame(self, idx: Optional[int] = None) -> np.ndarray:
        if idx is None:
            idx = self.current_frame
        idx = max(0, min(idx, self.total - 1))
        return self.frames[idx]

# ==== 描画関数（最適化版） ====
def draw_video(screen: pygame.Surface, frame: np.ndarray, dx: int = 0, dy: int = 0):
    """動画フレーム描画"""
    config = DisplayConfig()
    frame_resized = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
    surf = pygame.image.frombuffer(
        frame_resized.tobytes(),
        (frame_resized.shape[1], frame_resized.shape[0]),
        "RGBA"
    ).convert_alpha()
    screen.blit(surf, (dx, dy))

def draw_video_fullscreen(screen: pygame.Surface, video: AlphaVideo):
    """フルスクリーン動画描画"""
    draw_video(screen, video.get_frame(), 0, 0)

def draw_subtitle(screen: pygame.Surface, font: pygame.font.Font):
    """パンプキンの応答字幕（下部）"""
    with g_state.subtitle_lock:
        text = g_state.current_subtitle
    
    if not text:
        return
    
    config = DisplayConfig()
    sub_config = SubtitleConfig()
    
    # テキスト折り返し
    lines = []
    current_line = ""
    
    for char in text:
        test_line = current_line + char
        if font.size(test_line)[0] > sub_config.MAX_WIDTH:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # レンダリング
    rendered_lines = [font.render(line, True, sub_config.COLOR) for line in lines]
    max_width = max(surf.get_width() for surf in rendered_lines)
    total_height = sum(surf.get_height() + 5 for surf in rendered_lines)
    
    # 背景描画
    padding = 20
    y_position = config.HEIGHT - 150
    bg_rect = pygame.Rect(
        (config.WIDTH - max_width - padding * 2) // 2,
        y_position - padding,
        max_width + padding * 2,
        total_height + padding * 2
    )
    
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    bg_surface.fill(sub_config.BG_COLOR)
    screen.blit(bg_surface, bg_rect)
    
    # テキスト描画
    y_offset = y_position
    for surf in rendered_lines:
        x = (config.WIDTH - surf.get_width()) // 2
        screen.blit(surf, (x, y_offset))
        y_offset += surf.get_height() + 5

def draw_user_subtitle(screen: pygame.Surface, font: pygame.font.Font):
    """ユーザー質問字幕（上部・通知風）"""
    with g_state.user_subtitle_lock:
        if not g_state.user_subtitle_active:
            return
        
        text = g_state.user_subtitle_text
        elapsed = time.time() - g_state.user_subtitle_start_time
        
        if elapsed > SubtitleConfig.USER_DISPLAY_TIME + SubtitleConfig.USER_FADE_DURATION:
            g_state.user_subtitle_active = False
            return
    
    config = DisplayConfig()
    sub_config = SubtitleConfig()
    
    # テキスト折り返し
    lines = []
    current_line = ""
    
    for char in text:
        test_line = current_line + char
        if font.size(test_line)[0] > sub_config.USER_MAX_WIDTH:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # レンダリング
    rendered_lines = [font.render(line, True, sub_config.USER_COLOR) for line in lines]
    max_width = max(surf.get_width() for surf in rendered_lines)
    total_height = sum(surf.get_height() + 5 for surf in rendered_lines)
    
    padding = 20
    corner_radius = 15
    
    # アニメーション計算
    y_pos = 20.0
    alpha = 255
    
    if elapsed < sub_config.USER_SLIDE_DURATION:
        progress = elapsed / sub_config.USER_SLIDE_DURATION
        progress = 1 - pow(1 - progress, 3)  # easeOutCubic
        y_pos = -total_height - padding * 2 + (total_height + padding * 2 + 20) * progress
    elif elapsed > sub_config.USER_DISPLAY_TIME:
        fade_progress = (elapsed - sub_config.USER_DISPLAY_TIME) / sub_config.USER_FADE_DURATION
        alpha = int(255 * (1 - fade_progress))
    
    # 背景描画
    bg_rect = pygame.Rect(
        (config.WIDTH - max_width - padding * 2) // 2,
        int(y_pos),
        max_width + padding * 2,
        total_height + padding * 2
    )
    
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    pygame.draw.rect(
        bg_surface,
        (*sub_config.USER_BG_COLOR[:3], min(sub_config.USER_BG_COLOR[3], alpha)),
        (0, 0, bg_rect.width, bg_rect.height),
        border_radius=corner_radius
    )
    screen.blit(bg_surface, bg_rect)
    
    # テキスト描画
    y_offset = int(y_pos) + padding
    for surf in rendered_lines:
        surf = surf.copy()
        surf.set_alpha(alpha)
        x = (config.WIDTH - surf.get_width()) // 2
        screen.blit(surf, (x, y_offset))
        y_offset += surf.get_height() + 5

# ==== おしゃべりモーション ====
def handle_a_key_video(vid: AlphaVideo, t: int, seed: int = 0) -> Tuple[np.ndarray, int, int]:
    """Aキー連動動画処理"""
    dx, dy = float_motion(t, seed=seed, amp_y=22, amp_x=12, base_speed=0.0009)

    if g_state.a_key_active:
        vid.current_frame += vid.direction
        
        if vid.current_frame >= vid.total - 1:
            vid.current_frame = vid.total - 1
            vid.direction = -1
        elif vid.current_frame <= 0:
            vid.current_frame = 0
            vid.direction = 1
    else:
        vid.current_frame = 0
        vid.direction = 1

    return vid.get_frame(), dx, dy

# ==== Flask API ====
app_text = Flask(__name__ + '_text')
pumpkin_talk: Optional[PumpkinTalk] = None

@app_text.route('/receive_text', methods=['POST'])
def receive_text():
    data = request.get_json()
    input_text = data.get("text", "")
    
    if not input_text:
        return jsonify({"status": "error"}), 400
    
    if g_state.state not in [State.IDLE, State.ENTRY, State.FINISH]:
        threading.Thread(
            target=pumpkin_talk.process_input_text,
            args=(input_text,),
            daemon=True
        ).start()
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "ignored"}), 200

app_key = Flask(__name__ + '_key')

@app_key.route('/key_event', methods=['POST'])
def receive_key_event():
    data = request.get_json()
    key_name = data.get("key")

    key_map = {
        'left': K_LEFT,
        'right': K_RIGHT,
        'a': K_a,
        'k': K_k,
        'l': K_l,
        'q': K_q
    }
    
    if key_name in key_map:
        if key_name == 'q':
            with g_state.skip_lock:
                g_state.skip_flag = True
            print("[🛑 緊急スキップ受信]")
        pygame.event.post(pygame.event.Event(KEYDOWN, key=key_map[key_name]))
    elif key_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        pygame.event.post(pygame.event.Event(USEREVENT, key=key_name))
    else:
        return jsonify({"status": "error"}), 400

    return jsonify({"status": "success"})

app_monitor = Flask(__name__ + '_monitor')

@app_monitor.route('/get_response', methods=['GET'])
def get_response():
    """最新のAI応答を返す"""
    with g_state.response_lock:
        return jsonify({"response": g_state.latest_response})

# ==== メイン処理 ====
def main():
    global pumpkin_talk
    
    print("初期化中...")
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    
    pygame.init()
    config = DisplayConfig()
    screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
    pygame.display.set_caption("AI_pumpkin_talk")
    clock = pygame.time.Clock()
    
    # フォント読み込み
    try:
        font = pygame.font.Font("ZenKakuGothicNew-Regular.ttf", SubtitleConfig.FONT_SIZE)
        user_font = pygame.font.Font("ZenKakuGothicNew-Regular.ttf", SubtitleConfig.USER_FONT_SIZE)
    except:
        font = pygame.font.Font(None, SubtitleConfig.FONT_SIZE)
        user_font = pygame.font.Font(None, SubtitleConfig.USER_FONT_SIZE)

    # 背景動画
    cap_bg = cv2.VideoCapture(config.BG_VIDEO_PATH)
    if not cap_bg.isOpened():
        print("[ERROR] 背景動画が開けません")
        sys.exit()
    
    ret_bg, frame_bg = cap_bg.read()
    if not ret_bg:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, frame_bg = cap_bg.read()
    bg_counter = 0

    # キャラクター動画
    video_config = VideoConfig()
    videos = {
        "normal": AlphaVideo(video_config.MAIN),
        "full2": AlphaVideo(video_config.FULL2),
        "full3": AlphaVideo(video_config.FULL3),
        "full4": AlphaVideo(video_config.FULL4),
        "full5": AlphaVideo(video_config.FULL5),
        "full6": AlphaVideo(video_config.FULL6),
        "full7": AlphaVideo(video_config.FULL7),
        "entry": AlphaVideo(video_config.ENTRY),
        "finish": AlphaVideo(video_config.FINISH)
    }

    g_state.state = State.IDLE
    float_offset_x = 0.0
    float_offset_y = 0.0
    transition_blend = 1.0

    # Flaskサーバー起動
    threading.Thread(
        target=lambda: app_text.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True),
        daemon=True
    ).start()
    
    threading.Thread(
        target=lambda: app_key.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True),
        daemon=True
    ).start()
    
    threading.Thread(
        target=lambda: app_monitor.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False, threaded=True),
        daemon=True
    ).start()
    
    time.sleep(1)
    print("起動完了(ストリーミングモード + 緊急スキップ対応)")

    # メインループ
    while True:
        t = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_a:
                with g_state.audio_lock:
                    g_state.a_key_active = not g_state.a_key_active
            elif event.type == KEYDOWN:
                if event.key == K_k:
                    if g_state.state == State.IDLE:
                        g_state.state = State.ENTRY
                        videos["entry"].current_frame = 0
                        videos["entry"].frame_accumulator = 0.0
                elif event.key == K_l:
                    if g_state.state in [State.NORMAL, State.FULL3, State.FULL6]:
                        g_state.state = State.FINISH
                        videos["finish"].current_frame = 0
                        videos["finish"].frame_accumulator = 0.0
                elif event.key == K_LEFT:
                    if g_state.state == State.NORMAL:
                        g_state.state = State.FULL2
                    elif g_state.state == State.FULL3:
                        g_state.state = State.FULL4
                elif event.key == K_RIGHT:
                    if g_state.state == State.NORMAL:
                        g_state.state = State.FULL5
                    elif g_state.state == State.FULL6:
                        g_state.state = State.FULL7
                elif event.key == K_q:
                    with g_state.skip_lock:
                        g_state.skip_flag = True
                    print("[🛑 ローカルスキップ実行]")
            elif event.type == USEREVENT:
                key_num = event.key
                if key_num in SOUND_FILES:
                    wav_path = SOUND_FILES[key_num]
                    if os.path.exists(wav_path):
                        def play_sound():
                            pumpkin_talk.play_audio_with_aplay(wav_path)
                        threading.Thread(target=play_sound, daemon=True).start()

        # 背景描画
        if bg_counter % config.BACK_SPEED_SKIP == 0:
            ret_bg, frame_bg = cap_bg.read()
            if not ret_bg:
                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg, frame_bg = cap_bg.read()
        bg_counter += 1
        
        bg_rgb = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
        bg_rgb = cv2.resize(bg_rgb, (config.WIDTH, config.HEIGHT))
        bg_surf = pygame.image.frombuffer(
            bg_rgb.tobytes(),
            (bg_rgb.shape[1], bg_rgb.shape[0]),
            "RGB"
        )
        screen.blit(bg_surf, (0, 0))

        # 浮遊モーション計算
        base_float_x, base_float_y = float_motion(t, seed=1)
        is_transitioning = g_state.state in [State.FULL2, State.FULL4, State.FULL5, State.FULL7]
        
        if is_transitioning:
            transition_blend = max(0.0, transition_blend - 0.05)
        else:
            transition_blend = min(1.0, transition_blend + 0.05)
        
        float_offset_x = base_float_x * transition_blend
        float_offset_y = base_float_y * transition_blend
        
        dx = int(float_offset_x)
        dy = int(float_offset_y)

        # 状態管理
        if g_state.state == State.IDLE:
            pass
            
        elif g_state.state == State.ENTRY:
            draw_video_fullscreen(screen, videos["entry"])
            videos["entry"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["entry"].frame_accumulator >= 1.0:
                videos["entry"].current_frame += int(videos["entry"].frame_accumulator)
                videos["entry"].frame_accumulator -= int(videos["entry"].frame_accumulator)
            if videos["entry"].current_frame >= videos["entry"].total:
                videos["entry"].current_frame = 0
                videos["entry"].frame_accumulator = 0.0
                g_state.state = State.NORMAL
                transition_blend = 0.0
                
        elif g_state.state == State.FINISH:
            draw_video_fullscreen(screen, videos["finish"])
            videos["finish"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["finish"].frame_accumulator >= 1.0:
                videos["finish"].current_frame += int(videos["finish"].frame_accumulator)
                videos["finish"].frame_accumulator -= int(videos["finish"].frame_accumulator)
            if videos["finish"].current_frame >= videos["finish"].total:
                videos["finish"].current_frame = 0
                videos["finish"].frame_accumulator = 0.0
                g_state.state = State.IDLE
                
        elif g_state.state == State.NORMAL:
            frame, _, _ = handle_a_key_video(videos["normal"], t, seed=1)
            draw_video(screen, frame, dx, dy)
            
        elif g_state.state == State.FULL3:
            frame, _, _ = handle_a_key_video(videos["full3"], t, seed=3)
            draw_video(screen, frame, dx, dy)
            
        elif g_state.state == State.FULL6:
            frame, _, _ = handle_a_key_video(videos["full6"], t, seed=5)
            draw_video(screen, frame, dx, dy)
            
        elif g_state.state == State.FULL2:
            frame = videos["full2"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full2"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["full2"].frame_accumulator >= 1.0:
                videos["full2"].current_frame += int(videos["full2"].frame_accumulator)
                videos["full2"].frame_accumulator -= int(videos["full2"].frame_accumulator)
            if videos["full2"].current_frame >= videos["full2"].total:
                videos["full2"].current_frame = 0
                videos["full2"].frame_accumulator = 0.0
                g_state.state = State.FULL3
                transition_blend = 0.0
                
        elif g_state.state == State.FULL4:
            frame = videos["full4"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full4"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["full4"].frame_accumulator >= 1.0:
                videos["full4"].current_frame += int(videos["full4"].frame_accumulator)
                videos["full4"].frame_accumulator -= int(videos["full4"].frame_accumulator)
            if videos["full4"].current_frame >= videos["full4"].total:
                videos["full4"].current_frame = 0
                videos["full4"].frame_accumulator = 0.0
                g_state.state = State.NORMAL
                transition_blend = 0.0
                
        elif g_state.state == State.FULL5:
            frame = videos["full5"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full5"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["full5"].frame_accumulator >= 1.0:
                videos["full5"].current_frame += int(videos["full5"].frame_accumulator)
                videos["full5"].frame_accumulator -= int(videos["full5"].frame_accumulator)
            if videos["full5"].current_frame >= videos["full5"].total:
                videos["full5"].current_frame = 0
                videos["full5"].frame_accumulator = 0.0
                g_state.state = State.FULL6
                transition_blend = 0.0
                
        elif g_state.state == State.FULL7:
            frame = videos["full7"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full7"].frame_accumulator += video_config.TRANSITION_SPEED
            if videos["full7"].frame_accumulator >= 1.0:
                videos["full7"].current_frame += int(videos["full7"].frame_accumulator)
                videos["full7"].frame_accumulator -= int(videos["full7"].frame_accumulator)
            if videos["full7"].current_frame >= videos["full7"].total:
                videos["full7"].current_frame = 0
                videos["full7"].frame_accumulator = 0.0
                g_state.state = State.NORMAL
                transition_blend = 0.0

        # 字幕描画
        draw_subtitle(screen, font)
        draw_user_subtitle(screen, user_font)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()