# -*- coding: utf-8 -*-
# ストリーミング対応版 1.py - 字幕機能強化版

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

# ==== 映像設定 ====
w, h = 1920, 1020
BG_VIDEO_PATH = "videos/BG.mp4"
VIDEO_MAIN = "videos/Pumpkin-Center.mov"
VIDEO_FULL2 = "videos/Pumpkin-Center2Left.mov"
VIDEO_FULL3 = "videos/Pumpkin-Left.mov"
VIDEO_FULL4 = "videos/Pumpkin-Left2Center.mov"
VIDEO_FULL5 = "videos/Pumpkin-Center2Right.mov"
VIDEO_FULL6 = "videos/Pumpkin-Right.mov"
VIDEO_FULL7 = "videos/Pumpkin-Right2Center.mov"
VIDEO_ENTRY = "videos/Pumpkin-Entry.mov"
VIDEO_FINISH = "videos/Pumpkin-Finish.mov"
BACK_SPEED_SKIP = 10
TRANSITION_SPEED = 1.0

# ==== 字幕設定（パンプキンの応答） ====
SUBTITLE_FONT_SIZE = 48
SUBTITLE_COLOR = (255, 255, 255)
SUBTITLE_BG_COLOR = (0, 0, 0, 180)
SUBTITLE_Y_POSITION = h - 150
SUBTITLE_MAX_WIDTH = w - 200
CHAR_DURATION = 0.13
PUNCTUATION_DURATION = 0.43

# ==== ユーザー質問字幕設定 ====
USER_SUBTITLE_FONT_SIZE = 36
USER_SUBTITLE_COLOR = (255, 255, 255)
USER_SUBTITLE_BG_COLOR = (40, 40, 40, 220)
USER_SUBTITLE_MAX_WIDTH = w - 400
USER_SUBTITLE_DISPLAY_TIME = 4.0  # 表示時間（秒）
USER_SUBTITLE_SLIDE_DURATION = 0.3  # スライドインアニメーション時間
USER_SUBTITLE_FADE_DURATION = 0.4  # フェードアウト時間

SOUND_FILES = {
    '1': "sounds/OP1.wav",
    '2': "sounds/sound2.wav",
    '3': "sounds/sound3.wav",
    '4': "sounds/sound4.wav",
    '5': "sounds/sound5.wav",
    '6': "sounds/sound6.wav",
    '7': "sounds/sound7.wav",
    '8': "sounds/sound8.wav",
    '9': "sounds/sound9.wav",
    '0': "sounds/sound0.wav"
}

# グローバル変数
a_key_active = False
state = "idle"
audio_lock = threading.Lock()
current_subtitle = ""
subtitle_lock = threading.Lock()
is_speaking = False

# ユーザー質問字幕用
user_subtitle_text = ""
user_subtitle_start_time = 0
user_subtitle_active = False
user_subtitle_lock = threading.Lock()

speaking_lock = threading.Lock()

# HTTPセッション（接続プーリング）
session = requests.Session()
session.mount('http://', requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))

# ==== Config Loader ====
class LoadConfig:
    def __init__(self, config_path="pumpkin.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def get_character_prompt(self):
        char = self.config["character"]
        knowledge_dict = self.config.get("knowledge", {})
        knowledge_str = ""
        for category, items in knowledge_dict.items():
            knowledge_str += f"\n【{category}】\n" + "\n".join(items) + "\n"
        return char["prompt"].format(knowledge=knowledge_str)
    
    def get_ollama_config(self):
        return self.config["api"]["ollama"]
    
    def get_voicevox_config(self):
        return self.config["api"]["voicevox"]
    
    def get_system_config(self):
        return self.config.get("system", {})
    
    def get_advanced_config(self):
        return self.config.get("advanced", {})

# ==== PumpkinTalk ====
class PumpkinTalk:
    def __init__(self, config_path="pumpkin.json"):
        self.config_loader = LoadConfig(config_path)
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
        
        # VOICEVOXキャッシュ設定
        self.voicevox_settings = self.system_config.get("voicevox", {})

    def split_sentences(self, text):
        """テキストを文単位に分割"""
        sentences = re.split(r'([。！？!?])', text)
        result = []
        temp = ""
        
        for i, part in enumerate(sentences):
            temp += part
            if part in ['。', '！', '？', '!', '?']:
                result.append(temp.strip())
                temp = ""
        
        if temp.strip():
            result.append(temp.strip())
        
        return [s for s in result if s]

    def generate_response_streaming(self, input_text):
        """ストリーミングで応答生成"""
        if not input_text:
            yield "何か言ったか?もう一度言ってみろよ!"
            return
        
        try:
            self.conversation_history.append(f"ユーザー: {input_text}")
            recent_history = "\n".join(self.conversation_history[-3:])
            
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": True,
                "options": self.ollama_config.get("params", {})
            }
            
            response = session.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            buffer = ""
            full_response = ""
            
            for line in response.iter_lines():
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
            
        except Exception as e:
            print(f"Ollama APIエラー: {e}")
            yield "ちっ、調子が悪いぞ!もう一度話しかけてみろよ!"

    def text_to_speech_fast(self, text):
        """高速音声合成"""
        try:
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = session.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            if self.voicevox_settings:
                query_data.update({
                    "speedScale": self.voicevox_settings.get("speed", 1.3),
                    "pitchScale": self.voicevox_settings.get("pitch", 0.0),
                    "intonationScale": self.voicevox_settings.get("intonation", 1.0),
                    "volumeScale": self.voicevox_settings.get("volume", 1.0),
                    "postPhonemeLength": self.voicevox_settings.get("post_phoneme_length", 0.2)
                })
            
            synthesis_url = f"{self.voicevox_url}/synthesis"
            synthesis_params = {"speaker": self.speaker_id}
            synthesis_response = session.post(
                synthesis_url, 
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            synthesis_response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(synthesis_response.content)
                return tmp.name
            
        except Exception as e:
            print(f"VOICEVOX APIエラー: {e}")
            return None

    def get_audio_duration(self, wav_file):
        try:
            with wave.open(wav_file, 'rb') as wf:
                return wf.getnframes() / float(wf.getframerate())
        except:
            return 0

    def display_subtitle_gradually(self, text, duration):
        global current_subtitle
        
        if duration <= 0:
            duration = 3.0
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in ["。", "!", "?"]:
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
        
        if not sentences:
            sentences = [text]
        
        for sentence in sentences:
            char_count = len(sentence)
            comma_count = sentence.count("、")
            period_count = sentence.count("。")
            tcomma_count = sentence.count("...")
            mark_count = sentence.count("!") + sentence.count("?") + sentence.count("*")
            char_count -= mark_count
            
            sentence_duration = (
                char_count * CHAR_DURATION + 
                (comma_count + period_count + tcomma_count) * PUNCTUATION_DURATION
            )
            
            with subtitle_lock:
                current_subtitle = sentence
            
            time.sleep(sentence_duration)
            
            with subtitle_lock:
                current_subtitle = ""

    def play_audio_with_aplay(self, wav_file, show_subtitle=False, subtitle_text="", is_final=False):
        global a_key_active, is_speaking
        
        if not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
            return

        try:
            with speaking_lock:
                if not is_speaking:
                    is_speaking = True
                    with audio_lock:
                        a_key_active = True
            
            duration = self.get_audio_duration(wav_file)
            
            if show_subtitle and subtitle_text:
                threading.Thread(
                    target=self.display_subtitle_gradually,
                    args=(subtitle_text, duration),
                    daemon=True
                ).start()
            
            subprocess.run(["aplay", "-q", wav_file], check=False)
            
            try:
                os.unlink(wav_file)
            except:
                pass
                
        except Exception as e:
            print(f"音声再生エラー: {e}")
        finally:
            if is_final:
                with speaking_lock:
                    is_speaking = False
                with audio_lock:
                    a_key_active = False

    def process_input_text(self, input_text):
        """ストリーミング処理"""
        print(f"受信: {input_text}")
        
        # ユーザー質問字幕を表示
        show_user_subtitle(input_text)
        
        sentences = list(self.generate_response_streaming(input_text))
        total = len(sentences)
        
        for idx, sentence in enumerate(sentences):
            if sentence:
                print(f"生成: {sentence}")
                wav_file = self.text_to_speech_fast(sentence)
                
                if wav_file:
                    is_final = (idx == total - 1)
                    self.play_audio_with_aplay(wav_file, show_subtitle=True, subtitle_text=sentence, is_final=is_final)

    def filter_response(self, response_text):
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()

# ==== ユーザー質問字幕表示関数 ====
def show_user_subtitle(text):
    """ユーザーの質問を上部に表示"""
    global user_subtitle_text, user_subtitle_start_time, user_subtitle_active
    
    with user_subtitle_lock:
        user_subtitle_text = text
        user_subtitle_start_time = time.time()
        user_subtitle_active = True

# ==== 浮遊モーション ====
def float_motion(t, seed=0, amp_y=16, amp_x=7, base_speed=0.0007):
    dy = math.sin(t * base_speed + seed) * amp_y
    dx = (
        math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x
        + math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6
        + math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
    )
    return int(dx), int(dy)

# ==== PyAV動画クラス ====
class AlphaVideo:
    def __init__(self, path):
        container = av.open(path)
        stream = container.streams.video[0]
        self.frames = []
        for frame in container.decode(stream):
            rgba = frame.to_ndarray(format="rgba")
            self.frames.append(rgba)
        self.total = len(self.frames)
        self.current_frame = 0
        self.direction = 1
        self.frame_accumulator = 0.0

    def get_frame(self, idx=None):
        if idx is None:
            idx = self.current_frame
        idx = max(0, min(idx, self.total-1))
        return self.frames[idx]

# ==== 描画関数 ====
def draw_video(screen, frame, dx=0, dy=0):
    frame_resized = cv2.resize(frame, (w, h))
    surf = pygame.image.frombuffer(frame_resized.tobytes(), frame_resized.shape[1::-1], "RGBA")
    surf = surf.convert_alpha()
    screen.blit(surf, (dx, dy))

def draw_video_fullscreen(screen, video):
    frame = video.get_frame()
    draw_video(screen, frame, 0, 0)

def draw_subtitle(screen, font):
    """パンプキンの応答字幕（下部）"""
    global current_subtitle
    
    with subtitle_lock:
        text = current_subtitle
    
    if not text:
        return
    
    lines = []
    current_line = ""
    
    for char in text:
        test_line = current_line + char
        test_surface = font.render(test_line, True, SUBTITLE_COLOR)
        if test_surface.get_width() > SUBTITLE_MAX_WIDTH:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    max_width = 0
    total_height = 0
    rendered_lines = []
    
    for line in lines:
        rendered = font.render(line, True, SUBTITLE_COLOR)
        rendered_lines.append(rendered)
        max_width = max(max_width, rendered.get_width())
        total_height += rendered.get_height() + 5
    
    padding = 20
    bg_rect = pygame.Rect(
        (w - max_width - padding * 2) // 2,
        SUBTITLE_Y_POSITION - padding,
        max_width + padding * 2,
        total_height + padding * 2
    )
    
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    bg_surface.fill(SUBTITLE_BG_COLOR)
    screen.blit(bg_surface, bg_rect)
    
    y_offset = SUBTITLE_Y_POSITION
    for rendered in rendered_lines:
        x = (w - rendered.get_width()) // 2
        screen.blit(rendered, (x, y_offset))
        y_offset += rendered.get_height() + 5

def draw_user_subtitle(screen, font):
    """ユーザー質問字幕（上部・iPhone通知風）"""
    global user_subtitle_text, user_subtitle_start_time, user_subtitle_active
    
    with user_subtitle_lock:
        if not user_subtitle_active:
            return
        
        text = user_subtitle_text
        elapsed = time.time() - user_subtitle_start_time
        
        # 表示時間を超えたら非表示
        if elapsed > USER_SUBTITLE_DISPLAY_TIME + USER_SUBTITLE_FADE_DURATION:
            user_subtitle_active = False
            return
    
    # テキストを折り返し
    lines = []
    current_line = ""
    
    for char in text:
        test_line = current_line + char
        test_surface = font.render(test_line, True, USER_SUBTITLE_COLOR)
        if test_surface.get_width() > USER_SUBTITLE_MAX_WIDTH:
            if current_line:
                lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line)
    
    # 描画サイズ計算
    max_width = 0
    total_height = 0
    rendered_lines = []
    
    for line in lines:
        rendered = font.render(line, True, USER_SUBTITLE_COLOR)
        rendered_lines.append(rendered)
        max_width = max(max_width, rendered.get_width())
        total_height += rendered.get_height() + 5
    
    padding = 20
    corner_radius = 15
    
    # アニメーション計算
    y_pos = 20  # 上部からの距離
    alpha = 255
    
    # スライドインアニメーション
    if elapsed < USER_SUBTITLE_SLIDE_DURATION:
        progress = elapsed / USER_SUBTITLE_SLIDE_DURATION
        # easeOutCubic
        progress = 1 - pow(1 - progress, 3)
        y_pos = -total_height - padding * 2 + (total_height + padding * 2 + 20) * progress
    
    # フェードアウトアニメーション
    elif elapsed > USER_SUBTITLE_DISPLAY_TIME:
        fade_progress = (elapsed - USER_SUBTITLE_DISPLAY_TIME) / USER_SUBTITLE_FADE_DURATION
        alpha = int(255 * (1 - fade_progress))
    
    # 背景描画（角丸）
    bg_rect = pygame.Rect(
        (w - max_width - padding * 2) // 2,
        int(y_pos),
        max_width + padding * 2,
        total_height + padding * 2
    )
    
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    
    # 角丸矩形描画
    pygame.draw.rect(bg_surface, (*USER_SUBTITLE_BG_COLOR[:3], min(USER_SUBTITLE_BG_COLOR[3], alpha)), 
                     (0, 0, bg_rect.width, bg_rect.height), border_radius=corner_radius)
    
    screen.blit(bg_surface, bg_rect)
    
    # テキスト描画
    y_offset = int(y_pos) + padding
    for rendered in rendered_lines:
        text_surface = rendered.copy()
        text_surface.set_alpha(alpha)
        x = (w - rendered.get_width()) // 2
        screen.blit(text_surface, (x, y_offset))
        y_offset += rendered.get_height() + 5

# ==== おしゃべりモーション ====
def handle_a_key_video(vid, t, seed=0):
    dx, dy = float_motion(t, seed=seed, amp_y=22, amp_x=12, base_speed=0.0009)

    if a_key_active:
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

# ==== Flask ====
app_text = Flask(__name__ + '_text')
pumpkin_talk = None

@app_text.route('/receive_text', methods=['POST'])
def receive_text():
    global state
    data = request.get_json()
    input_text = data.get("text", "")
    if input_text:
        if state not in ["idle", "entry", "finish"]:
            threading.Thread(target=pumpkin_talk.process_input_text, args=(input_text,), daemon=True).start()
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "ignored"}), 200
    else:
        return jsonify({"status": "error"}), 400

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
        'l': K_l
    }
    
    if key_name in key_map:
        pygame.event.post(pygame.event.Event(KEYDOWN, key=key_map[key_name]))
    elif key_name in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
        pygame.event.post(pygame.event.Event(USEREVENT, key=key_name))
    else:
        return jsonify({"status": "error"}), 400

    return jsonify({"status": "success"})

# ==== メイン ====
def main():
    global state, a_key_active, pumpkin_talk
    
    print("初期化中...")
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("AI_pumpkin_talk")
    clock = pygame.time.Clock()
    
    try:
        font = pygame.font.Font("ZenKakuGothicNew-Regular.ttf", SUBTITLE_FONT_SIZE)
        user_font = pygame.font.Font("ZenKakuGothicNew-Regular.ttf", USER_SUBTITLE_FONT_SIZE)
    except:
        font = pygame.font.Font(None, SUBTITLE_FONT_SIZE)
        user_font = pygame.font.Font(None, USER_SUBTITLE_FONT_SIZE)

    cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
    if not cap_bg.isOpened():
        sys.exit()
    ret_bg, frame_bg = cap_bg.read()
    if not ret_bg:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, frame_bg = cap_bg.read()
    bg_counter = 0

    videos = {
        "normal": AlphaVideo(VIDEO_MAIN),
        "full2": AlphaVideo(VIDEO_FULL2),
        "full3": AlphaVideo(VIDEO_FULL3),
        "full4": AlphaVideo(VIDEO_FULL4),
        "full5": AlphaVideo(VIDEO_FULL5),
        "full6": AlphaVideo(VIDEO_FULL6),
        "full7": AlphaVideo(VIDEO_FULL7),
        "entry": AlphaVideo(VIDEO_ENTRY),
        "finish": AlphaVideo(VIDEO_FINISH)
    }

    state = "idle"
    float_offset_x = 0.0
    float_offset_y = 0.0
    transition_blend = 1.0

    threading.Thread(
        target=lambda: app_text.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    ).start()
    
    threading.Thread(
        target=lambda: app_key.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False, threaded=True)
    ).start()
    
    time.sleep(1)
    print("起動完了(ストリーミングモード)")

    while True:
        t = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_a:
                with audio_lock:
                    a_key_active = not a_key_active
            elif event.type == KEYDOWN:
                if event.key == K_k:
                    if state == "idle":
                        state = "entry"
                        videos["entry"].current_frame = 0
                        videos["entry"].frame_accumulator = 0.0
                elif event.key == K_l:
                    if state in ["normal", "full3", "full6"]:
                        state = "finish"
                        videos["finish"].current_frame = 0
                        videos["finish"].frame_accumulator = 0.0
                elif event.key == K_LEFT:
                    if state == "normal": 
                        state = "full2"
                    elif state == "full3": 
                        state = "full4"
                elif event.key == K_RIGHT:
                    if state == "normal": 
                        state = "full5"
                    elif state == "full6": 
                        state = "full7"
            elif event.type == USEREVENT:
                key_num = event.key
                if key_num in SOUND_FILES:
                    wav_path = SOUND_FILES[key_num]
                    if os.path.exists(wav_path):
                        def play_sound():
                            pumpkin_talk.play_audio_with_aplay(wav_path)
                        threading.Thread(target=play_sound, daemon=True).start()

        # 背景描画
        if bg_counter % BACK_SPEED_SKIP == 0:
            ret_bg, frame_bg = cap_bg.read()
            if not ret_bg:
                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg, frame_bg = cap_bg.read()
        bg_counter += 1
        bg_rgb = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
        bg_rgb = cv2.resize(bg_rgb, (w, h))
        bg_surf = pygame.image.frombuffer(bg_rgb.tobytes(), bg_rgb.shape[1::-1], "RGB")
        screen.blit(bg_surf, (0, 0))

        base_float_x, base_float_y = float_motion(t, seed=1)
        is_transitioning = state in ["full2", "full4", "full5", "full7"]
        
        if is_transitioning:
            transition_blend = max(0.0, transition_blend - 0.05)
        else:
            transition_blend = min(1.0, transition_blend + 0.05)
        
        float_offset_x = base_float_x * transition_blend
        float_offset_y = base_float_y * transition_blend
        
        dx = int(float_offset_x)
        dy = int(float_offset_y)

        # 状態管理
        if state == "idle":
            pass
            
        elif state == "entry":
            draw_video_fullscreen(screen, videos["entry"])
            videos["entry"].frame_accumulator += TRANSITION_SPEED
            if videos["entry"].frame_accumulator >= 1.0:
                videos["entry"].current_frame += int(videos["entry"].frame_accumulator)
                videos["entry"].frame_accumulator -= int(videos["entry"].frame_accumulator)
            if videos["entry"].current_frame >= videos["entry"].total:
                videos["entry"].current_frame = 0
                videos["entry"].frame_accumulator = 0.0
                state = "normal"
                transition_blend = 0.0
                
        elif state == "finish":
            draw_video_fullscreen(screen, videos["finish"])
            videos["finish"].frame_accumulator += TRANSITION_SPEED
            if videos["finish"].frame_accumulator >= 1.0:
                videos["finish"].current_frame += int(videos["finish"].frame_accumulator)
                videos["finish"].frame_accumulator -= int(videos["finish"].frame_accumulator)
            if videos["finish"].current_frame >= videos["finish"].total:
                videos["finish"].current_frame = 0
                videos["finish"].frame_accumulator = 0.0
                state = "idle"
                
        elif state == "normal":
            frame, _, _ = handle_a_key_video(videos["normal"], t, seed=1)
            draw_video(screen, frame, dx, dy)
            
        elif state == "full3":
            frame, _, _ = handle_a_key_video(videos["full3"], t, seed=3)
            draw_video(screen, frame, dx, dy)
            
        elif state == "full6":
            frame, _, _ = handle_a_key_video(videos["full6"], t, seed=5)
            draw_video(screen, frame, dx, dy)
            
        elif state == "full2":
            frame = videos["full2"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full2"].frame_accumulator += TRANSITION_SPEED
            if videos["full2"].frame_accumulator >= 1.0:
                videos["full2"].current_frame += int(videos["full2"].frame_accumulator)
                videos["full2"].frame_accumulator -= int(videos["full2"].frame_accumulator)
            if videos["full2"].current_frame >= videos["full2"].total:
                videos["full2"].current_frame = 0
                videos["full2"].frame_accumulator = 0.0
                state = "full3"
                transition_blend = 0.0
                
        elif state == "full4":
            frame = videos["full4"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full4"].frame_accumulator += TRANSITION_SPEED
            if videos["full4"].frame_accumulator >= 1.0:
                videos["full4"].current_frame += int(videos["full4"].frame_accumulator)
                videos["full4"].frame_accumulator -= int(videos["full4"].frame_accumulator)
            if videos["full4"].current_frame >= videos["full4"].total:
                videos["full4"].current_frame = 0
                videos["full4"].frame_accumulator = 0.0
                state = "normal"
                transition_blend = 0.0
                
        elif state == "full5":
            frame = videos["full5"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full5"].frame_accumulator += TRANSITION_SPEED
            if videos["full5"].frame_accumulator >= 1.0:
                videos["full5"].current_frame += int(videos["full5"].frame_accumulator)
                videos["full5"].frame_accumulator -= int(videos["full5"].frame_accumulator)
            if videos["full5"].current_frame >= videos["full5"].total:
                videos["full5"].current_frame = 0
                videos["full5"].frame_accumulator = 0.0
                state = "full6"
                transition_blend = 0.0
                
        elif state == "full7":
            frame = videos["full7"].get_frame()
            draw_video(screen, frame, dx, dy)
            
            videos["full7"].frame_accumulator += TRANSITION_SPEED
            if videos["full7"].frame_accumulator >= 1.0:
                videos["full7"].current_frame += int(videos["full7"].frame_accumulator)
                videos["full7"].frame_accumulator -= int(videos["full7"].frame_accumulator)
            if videos["full7"].current_frame >= videos["full7"].total:
                videos["full7"].current_frame = 0
                videos["full7"].frame_accumulator = 0.0
                state = "normal"
                transition_blend = 0.0

        # 字幕描画（パンプキンの応答 - 下部）
        draw_subtitle(screen, font)
        
        # ユーザー質問字幕（上部 - iPhone通知風）
        draw_user_subtitle(screen, user_font)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()