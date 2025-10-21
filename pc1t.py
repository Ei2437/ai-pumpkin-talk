# -*- coding: utf-8 -*-
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

# ==== 設定 ====
w, h = 1920, 1020
BG_VIDEO_PATH = "video/BG.mp4"

VIDEO_MAIN = "video/Pumpkin-Center.mov"
VIDEO_FULL2 = "video/Pumpkin-Center2Left.mov"
VIDEO_FULL3 = "video/Pumpkin-Left.mov"
VIDEO_FULL4 = "video/Pumpkin-Left2Center.mov"
VIDEO_FULL5 = "video/Pumpkin-Center2Right.mov"
VIDEO_FULL6 = "video/Pumpkin-Right.mov"
VIDEO_FULL7 = "video/Pumpkin-Right2Center.mov"

BACK_SPEED_SKIP = 10  # 背景動画速度（フレーム単位）

# グローバル変数
a_key_active = False
state = "normal"

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

# ==== PumpkinTalk AI System ====
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
        
        self.temp_wav_file = "output.wav"

    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        try:
            self.conversation_history.append(f"ユーザー: {input_text}")
            recent_history = "\n".join(self.conversation_history[-6:])
            
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": False,
                "options": self.ollama_config.get("params", {})
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "応答を生成できませんでした。")
            
            if self.advanced_config:
                response_text = self.filter_response(response_text)
            
            self.conversation_history.append(f"パンプキン: {response_text}")
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぞ！もう一度話しかけてみろよ！"

    def text_to_speech(self, text):
        try:
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            if "voicevox" in self.system_config:
                voicevox_settings = self.system_config["voicevox"]
                if "speed" in voicevox_settings:
                    query_data["speedScale"] = voicevox_settings["speed"]
                if "pitch" in voicevox_settings:
                    query_data["pitchScale"] = voicevox_settings["pitch"]
                if "intonation" in voicevox_settings:
                    query_data["intonationScale"] = voicevox_settings["intonation"]
                if "volume" in voicevox_settings:
                    query_data["volumeScale"] = voicevox_settings["volume"]
                if "post_phoneme_length" in voicevox_settings:
                    query_data["postPhonemeLength"] = voicevox_settings["post_phoneme_length"]
            
            synthesis_url = f"{self.voicevox_url}/synthesis"
            synthesis_params = {"speaker": self.speaker_id}
            synthesis_response = requests.post(
                synthesis_url, 
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            synthesis_response.raise_for_status()
            
            wav_data = synthesis_response.content
            
            with open(self.temp_wav_file, "wb") as f:
                f.write(wav_data)
            print(f"音声ファイルを '{self.temp_wav_file}' に書き出しました")
            
            sample_rate, audio_data = wavfile.read(self.temp_wav_file)
            
            if len(audio_data.shape) == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None

    def play_audio_with_aplay(self):
        if not os.path.exists(self.temp_wav_file):
            print("再生できる音声ファイルがありません")
            return
        
        if os.path.getsize(self.temp_wav_file) == 0:
            print("警告: 音声ファイルが空です")
            return

        try:
            print(f"'{self.temp_wav_file}' を aplay で再生中...")
            result = subprocess.run(["aplay", "-q", self.temp_wav_file])
            if result.returncode == 0:
                print("再生完了")
            else:
                print(f"aplay でエラーが発生しました (終了コード: {result.returncode})")
        except FileNotFoundError:
            print("aplay が見つかりません。'sudo apt install alsa-utils' でインストールしてください。")
        except Exception as e:
            print(f"音声再生中にエラーが発生しました: {e}")

    def process_input_text(self, input_text):
        response_text = self.generate_response(input_text)
        print("回答:", response_text)
        
        print("音声合成中...")
        sample_rate, audio_data = self.text_to_speech(response_text)
        
        if sample_rate is not None and audio_data is not None:
            print("再生中...")
            self.play_audio_with_aplay()
        else:
            print("音声合成に失敗しました")

    def filter_response(self, response_text):
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            if "remove_patterns" in filtering:
                import re
                for pattern in filtering["remove_patterns"]:
                    response_text = re.sub(pattern, "", response_text)
            
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()

# ==== 浮遊モーション ====
def float_motion(t, seed=0, amp_y=16, amp_x=7, base_speed=0.0007):
    dy = math.sin(t * base_speed + seed) * amp_y
    dx = (
        math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x
        + math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6
        + math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
    )
    return int(dx), int(dy)

# ==== PyAV動画クラス（全フレーム先読み） ====
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

def draw_video_fullscreen(screen, video: AlphaVideo):
    frame = video.get_frame()
    draw_video(screen, frame, 0, 0)

# ==== Aキー往復再生（トグル対応） ====
def handle_a_key_video(vid: AlphaVideo, t, seed=0):
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

# ==== Flask App ====
app = Flask(__name__)
pumpkin_talk = None

@app.route('/receive_text', methods=['POST'])
def receive_text():
    data = request.get_json()
    input_text = data.get("text", "")
    if input_text:
        print(f"受信したテキスト: {input_text}")
        pumpkin_talk.process_input_text(input_text)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "error", "message": "No text provided"}), 400

@app.route('/key_event', methods=['POST'])
def receive_key_event():
    global a_key_active
    data = request.get_json()
    key_name = data.get("key")

    if key_name == 'left':
        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_LEFT))
        print("LEFT key event posted to pygame queue")
    elif key_name == 'right':
        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_RIGHT))
        print("RIGHT key event posted to pygame queue")
    elif key_name == 'a':
        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_a))
        print("A key event posted to pygame queue")
    else:
        print(f"Unknown key received: {key_name}")
        return jsonify({"status": "error", "message": "Unknown key"}), 400

    return jsonify({"status": "success"})

# ==== メイン ====
def main():
    global state, a_key_active, pumpkin_talk
    
    # PumpkinTalkの初期化
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    
    # Pygameの初期化
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("AI_pumpkin_talk")
    clock = pygame.time.Clock()

    # 背景動画
    cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
    if not cap_bg.isOpened():
        print("背景動画読み込み失敗")
        sys.exit()
    ret_bg, frame_bg = cap_bg.read()
    if not ret_bg:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, frame_bg = cap_bg.read()
    bg_counter = 0

    # 透過動画
    videos = {
        "normal": AlphaVideo(VIDEO_MAIN),
        "full2": AlphaVideo(VIDEO_FULL2),
        "full3": AlphaVideo(VIDEO_FULL3),
        "full4": AlphaVideo(VIDEO_FULL4),
        "full5": AlphaVideo(VIDEO_FULL5),
        "full6": AlphaVideo(VIDEO_FULL6),
        "full7": AlphaVideo(VIDEO_FULL7)
    }

    state = "normal"

    # Flaskサーバーを別スレッドで起動
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False))
    server_thread.daemon = True
    server_thread.start()
    print("Flaskサーバーを起動しました (port 5000)")

    while True:
        t = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_a:
                a_key_active = not a_key_active
                print(f"A key toggled: {'ON' if a_key_active else 'OFF'}")
            elif event.type == KEYDOWN:
                if event.key == K_LEFT:
                    if state == "normal": 
                        state = "full2"
                    elif state == "full3": 
                        state = "full4"
                elif event.key == K_RIGHT:
                    if state == "normal": 
                        state = "full5"
                    elif state == "full6": 
                        state = "full7"

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

        # 状態管理
        if state == "normal":
            frame, dx, dy = handle_a_key_video(videos["normal"], t, seed=1)
            draw_video(screen, frame, dx, dy)
        elif state == "full3":
            frame, dx, dy = handle_a_key_video(videos["full3"], t, seed=3)
            draw_video(screen, frame, dx, dy)
        elif state == "full6":
            frame, dx, dy = handle_a_key_video(videos["full6"], t, seed=5)
            draw_video(screen, frame, dx, dy)
        elif state == "full2":
            draw_video_fullscreen(screen, videos["full2"])
            videos["full2"].current_frame += 1
            if videos["full2"].current_frame >= videos["full2"].total:
                videos["full2"].current_frame = 0
                state = "full3"
        elif state == "full4":
            draw_video_fullscreen(screen, videos["full4"])
            videos["full4"].current_frame += 1
            if videos["full4"].current_frame >= videos["full4"].total:
                videos["full4"].current_frame = 0
                state = "normal"
        elif state == "full5":
            draw_video_fullscreen(screen, videos["full5"])
            videos["full5"].current_frame += 1
            if videos["full5"].current_frame >= videos["full5"].total:
                videos["full5"].current_frame = 0
                state = "full6"
        elif state == "full7":
            draw_video_fullscreen(screen, videos["full7"])
            videos["full7"].current_frame += 1
            if videos["full7"].current_frame >= videos["full7"].total:
                videos["full7"].current_frame = 0
                state = "normal"

        pygame.display.flip()
        clock.tick(60)

if __name__=="__main__":
    main()