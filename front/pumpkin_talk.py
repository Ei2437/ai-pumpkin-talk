# ver2.0  10/11 11:00 (pc1.py と video.py の統合版)

import os
import time
import json
import requests
import numpy as np
from scipy.io import wavfile
from flask import Flask, request, jsonify
import threading
import queue

# === 映像関連のインポート ===
import pygame
from pygame.locals import *
import sys
import cv2
import math
import random

# ==== 映像の設定 ====
# ---- 動画 ----
BG_VIDEO_PATH = "background.mp4"
VIDEO_MAIN = "Pumpkin_Center.mp4"
VIDEO_FULL2 = "Pumkin_Center2Left.mp4"
VIDEO_FULL4 = "Pumkin_Left2Center.mp4"
VIDEO_FULL3 = "Pumpkin_Left.mp4"
VIDEO_FULL5 = "Pumkin_Center2Right.mp4"
VIDEO_FULL6 = "Pumkin_Right.mp4"
VIDEO_FULL7 = "Pumkin_Right2Center.mp4"

#---- クロマキー処理の範囲 ----
LOWER_GREEN = np.array([20, 80, 80])
UPPER_GREEN = np.array([105, 255, 255])

#---- 画面サイズ ----
SCREEN_W = 1920 #横幅
SCREEN_H = 1020 #高さ

#---- 背景動画の再生速度 ----
BACK_GROUND_SPEED  = 10 #フレームで管理。10なら0.1倍速になる


class LoadConfig:
    # ... (pc1.py から変更なし) ...

class PumpkinTalk:
    def __init__(self, config_path="pumpkin.json"):
        # === 音声・AI関連の初期化 (pc1.py から) ===
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
        
        # 一時WAVファイル名
        self.temp_wav_file = "output.wav"

        # === 映像関連の初期化 (video.py から) ===
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("AI_pumpkin_talk")
        self.clock = pygame.time.Clock()

        # 背景動画を読み込み
        self.cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
        if not self.cap_bg.isOpened():
            print("背景動画の読み込みに失敗しました。")
            # sys.exit() は使わない。代わりにフラグを立てるか、例外を投げる
            raise RuntimeError("背景動画の読み込みに失敗しました。")
        
        # 背景動画のフレームを保持するための変数
        ret_bg, frame_bg = self.cap_bg.read()
        if not ret_bg:
            self.cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, frame_bg = self.cap_bg.read()
        self.frame_bg = frame_bg
        
        # 背景動画の再生速度制御用
        self.BG_SPEED_SKIP = BACK_GROUND_SPEED
        self.bg_frame_counter = 0

        self.cap_main = cv2.VideoCapture(VIDEO_MAIN)
        self.cap_full2 = cv2.VideoCapture(VIDEO_FULL2)
        self.cap_full3 = cv2.VideoCapture(VIDEO_FULL3)
        self.cap_full4 = cv2.VideoCapture(VIDEO_FULL4)
        self.cap_full5 = cv2.VideoCapture(VIDEO_FULL5)
        self.cap_full6 = cv2.VideoCapture(VIDEO_FULL6)
        self.cap_full7 = cv2.VideoCapture(VIDEO_FULL7)

        self.total_main = int(self.cap_main.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_f3 = int(self.cap_full3.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_f6 = int(self.cap_full6.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初期化
        self.state = "normal" # 映像の状態 ("normal", "talking", "full2", ...)
        self.dx_main = self.dy_main = self.dx_f3 = self.dy_f3 = self.dx_f6 = self.dy_f6 = 0
        self.current_main = self.current_f3 = self.current_f6 = 0.0
        self.move_start_time_main = self.move_start_time_f3 = self.move_start_time_f6 = 0
        self.start_frame_main = self.start_frame_f3 = self.start_frame_f6 = 0.0
        self.return_to_first_third_main = self.return_to_first_third_f3 = self.return_to_first_third_f6 = False
        
        # Pygameイベント処理用フラグ
        self.running = True
        
        # 音声再生状態監視用
        self.is_playing_audio = False

    # === 音声・AI関連メソッド (pc1.py から変更なしまたは軽微な変更) ===
    
    def generate_response(self, input_text):
        # ... (変更なし) ...

    def text_to_speech(self, text):
        # ... (変更なし) ...

    def filter_response(self, response_text):
        # ... (変更なし) ...

    # === 新規: pygame mixer で音声再生 ===
    def play_audio_with_pygame(self):
        """pygame mixer を使用して音声ファイルを再生"""
        if not os.path.exists(self.temp_wav_file):
            print("再生できる音声ファイルがありません")
            return False # 失敗
        
        if os.path.getsize(self.temp_wav_file) == 0:
            print("警告: 音声ファイルが空です")
            return False # 失敗

        try:
            print(f"'{self.temp_wav_file}' を pygame mixer で再生中...")
            pygame.mixer.init()
            pygame.mixer.music.load(self.temp_wav_file)
            pygame.mixer.music.play()
            
            # 再生状態をクラス変数で管理
            self.is_playing_audio = True
            
            # 再生が終わるまで待機
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100) # 少し待機
            
            pygame.mixer.quit()
            self.is_playing_audio = False # 再生終了
            print("再生完了")
            return True # 成功
        except Exception as e:
            print(f"pygame mixer での音声再生中にエラーが発生しました: {e}")
            self.is_playing_audio = False
            return False # 失敗

    # === 映像関連メソッド (video.py から移植・変更) ===
    
    def _chroma_key_rgba(self, frame_rgb, lower, upper):
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        mask = cv2.medianBlur(mask, 3)
        alpha = 255 - mask
        rgba = np.dstack((frame_rgb, alpha))
        return rgba

    def _float_motion(self, t, seed=0, amp_y=16, amp_x=7, base_speed=0.0007):
        dy = math.sin(t * base_speed + seed) * amp_y
        dx = (
            math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x
            + math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6
            + math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
        )
        return int(dx), int(dy)

    def _ease_in_out_sine(self, t):
        return -(math.cos(math.pi * t) - 1) / 2

    def _draw_video(self, screen, cap, size, dx, dy, lower, upper, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            rgba = self._chroma_key_rgba(frame, lower, upper)
            surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
            surf = surf.convert_alpha()
            screen.blit(surf, (dx, dy))

    def _draw_video_fullscreen(self, screen, frame, size, lower, upper):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        rgba = self._chroma_key_rgba(frame, lower, upper)
        surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
        surf = surf.convert_alpha()
        screen.blit(surf, (0, 0))

    def _update_background(self):
        """背景動画を更新する"""
        if self.bg_frame_counter % self.BG_SPEED_SKIP == 0:
            ret_bg_new, frame_bg_new = self.cap_bg.read()
            if not ret_bg_new:
                self.cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg_new, frame_bg_new = self.cap_bg.read()
            if ret_bg_new:
                self.frame_bg = frame_bg_new # 新しいフレームを保持
        
        bg_rgb = cv2.cvtColor(self.frame_bg, cv2.COLOR_BGR2RGB)
        bg_rgb = cv2.resize(bg_rgb, (SCREEN_W, SCREEN_H))
        bg_surface = pygame.image.frombuffer(bg_rgb.tobytes(), bg_rgb.shape[1::-1], "RGB")
        self.screen.blit(bg_surface, (0, 0))
        
        self.bg_frame_counter += 1

    def _handle_a_key(self, cap, total, current, start_frame, move_start_time, return_to_first_third, seed):
        """Aキー処理ロジック"""
        # 簡略化のため、ここではAキー処理を無効化または簡略化します。
        # 必要に応じて元のロジックを再実装してください。
        dx, dy = self._float_motion(pygame.time.get_ticks(), seed=seed, amp_y=22, amp_x=12, base_speed=0.0009)
        return dx, dy, current, start_frame, move_start_time, return_to_first_third

    def _draw_current_state(self):
        """現在の映像状態に応じて描画"""
        t = pygame.time.get_ticks()
        
        # === 各状態描画 ===
        if self.state == "normal":
            self.dx_main, self.dy_main, self.current_main, self.start_frame_main, self.move_start_time_main, self.return_to_first_third_main = \
                self._handle_a_key(self.cap_main, self.total_main, self.current_main, self.start_frame_main, self.move_start_time_main, self.return_to_first_third_main, seed=1)
            self._draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), self.dx_main, self.dy_main, LOWER_GREEN, UPPER_GREEN, self.current_main)
        
        elif self.state == "talking":
             # 例: talking 状態では、浮遊モーションを少し大きくするなど
             dx_talk, dy_talk = self._float_motion(t, seed=99, amp_y=25, amp_x=10, base_speed=0.001)
             self._draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), dx_talk, dy_talk, LOWER_GREEN, UPPER_GREEN, self.current_main)
             
        elif self.state == "full2":
            ret, frame = self.cap_full2.read()
            if not ret:
                self.cap_full3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.state = "full3"
            else:
                self._draw_video_fullscreen(self.screen, frame, (SCREEN_W, SCREEN_H), LOWER_GREEN, UPPER_GREEN)
        # ... (他の状態 "full3", "full4", ...) も同様に実装 ...
        # 今回は例として "normal" と "talking" と "full2" のみ実装
        
        else: # 未知の状態やデフォルト
             self._draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), self.dx_main, self.dy_main, LOWER_GREEN, UPPER_GREEN, self.current_main)

    def change_video_state(self, new_state):
        """映像の状態を変更する (外部から呼び出し可能)"""
        print(f"[PumpkinTalk] 映像状態変更: {self.state} -> {new_state}")
        if new_state in ["normal", "talking", "full2", "full3", "full4", "full5", "full6", "full7"]:
            self.state = new_state
        else:
            print(f"[PumpkinTalk] 無効な映像状態変更要求: {new_state}")

    def _handle_pygame_events(self):
        """Pygameのイベントを処理する"""
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self.running = False # メインループを停止するフラグを立てる

    def run_video_loop(self):
        """Pygameのメインループを実行する (別スレッドで実行)"""
        print("[PumpkinTalk] 映像ループを開始します...")
        while self.running:
            self._handle_pygame_events()
            self._update_background()
            self._draw_current_state()
            pygame.display.update()
            self.clock.tick(60) # 60 FPS
        
        # 終了処理
        for cap in [self.cap_main, self.cap_full2, self.cap_full3, self.cap_full4, self.cap_full5, self.cap_full6, self.cap_full7, self.cap_bg]:
            cap.release()
        pygame.quit()
        print("[PumpkinTalk] 映像ループを終了しました。")

    # === 音声・映像連携処理 ===
    
    def process_input_text(self, input_text):
        """音声応答と映像状態の連携処理"""
        
        # 1. 映像を "talking" 状態に変更
        self.change_video_state("talking")

        # 2. 応答の生成
        response_text = self.generate_response(input_text)
        print("回答:", response_text)
        
        # 3. 音声合成
        print("音声合成中...")
        sample_rate, audio_data = self.text_to_speech(response_text)
        
        # 4. 音声再生 (pygame mixer を使用)
        if sample_rate is not None and audio_data is not None:
            print("再生中...")
            success = self.play_audio_with_pygame() # self.is_playing_audio が更新される
            if not success:
                print("音声再生に失敗しました")
        else:
            print("音声合成に失敗しました")

        # 5. 再生が終わったら "normal" 状態に戻す
        #    (play_audio_with_pygame で self.is_playing_audio が False になる)
        self.change_video_state("normal")
        print("処理完了。")


def create_flask_app(pumpkin_talk_instance):
    """Flaskアプリケーションを作成する"""
    app = Flask(__name__)

    @app.route('/receive_text', methods=['POST'])
    def receive_text():
        data = request.get_json()
        input_text = data.get("text", "")
        if input_text:
            print(f"受信したテキスト: {input_text}")
            # 別スレッドで処理を実行して、即座にレスポンスを返す
            threading.Thread(target=pumpkin_talk_instance.process_input_text, args=(input_text,), daemon=True).start()
            return jsonify({"status": "processing started"}), 200
        else:
            return jsonify({"status": "error", "message": "No text provided"}), 400

    return app

def main():
    """メイン関数"""
    print("=== PumpkinTalk 統合版 (音声 + 映像) を起動します ===")
    
    # 1. PumpkinTalk インスタンスを作成
    try:
        pumpkin_talk = PumpkinTalk("pumpkin.json")
    except Exception as e:
        print(f"PumpkinTalk の初期化に失敗しました: {e}")
        return

    # 2. 映像ループを別スレッドで開始
    video_thread = threading.Thread(target=pumpkin_talk.run_video_loop, daemon=True)
    video_thread.start()
    print("映像スレッドを開始しました。")

    # 3. Flask サーバーを起動
    app = create_flask_app(pumpkin_talk)
    print("Flaskサーバーを起動します...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False) # use_reloader=False でスレッドの問題を回避

    # 4. Flaskサーバーが終了したら、映像スレッドも終了するようにフラグを立てる
    pumpkin_talk.running = False
    video_thread.join(timeout=5) # 最大5秒待つ
    print("=== PumpkinTalk を終了します ===")

if __name__ == "__main__":
    main()