# ver2.0  10/11 12:00 (pynput不要・Flask統合版)

import os
import sys
import time
import json
import threading
import queue
import math
import random
import cv2
import numpy as np
import pygame
from pygame.locals import *
from flask import Flask, request, jsonify
from scipy.io import wavfile

# ==== 設定 ====
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

# ==== プロキシ完全無効化 ====
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

REQUESTS_NO_PROXY = {"http": None, "https": None}
# =============================


# ==== クロマキー処理 ====
def chroma_key_rgba(frame_rgb, lower, upper):
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.medianBlur(mask, 3)
    alpha = 255 - mask
    rgba = np.dstack((frame_rgb, alpha))
    return rgba

# ==== ふわふわ浮遊モーション ====
def float_motion(t, seed=0, amp_y=16, amp_x=7, base_speed=0.0007):
    dy = math.sin(t * base_speed + seed) * amp_y
    dx = (
        math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x
        + math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6
        + math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
    )
    return int(dx), int(dy)

# ==== イージング ====
def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2

# ==== 中央に戻すアニメーション ====
def animate_to_center(current_offset, duration=300):
    start_time = pygame.time.get_ticks()
    start_dx, start_dy = current_offset
    while True:
        elapsed = pygame.time.get_ticks() - start_time
        if elapsed >= duration:
            break
        progress = elapsed / duration
        ease = ease_in_out_sine(progress)
        dx = int(start_dx * (1 - ease))
        dy = int(start_dy * (1 - ease))
        yield dx, dy
    yield 0, 0

# ==== 共通描画関数(クロマキー処理) ====
def draw_video(screen, cap, size, dx, dy, lower, upper, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        rgba = chroma_key_rgba(frame, lower, upper)
        surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
        surf = surf.convert_alpha()
        screen.blit(surf, (dx, dy))

def draw_video_fullscreen(screen, frame, size, lower, upper):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    rgba = chroma_key_rgba(frame, lower, upper)
    surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
    surf = surf.convert_alpha()
    screen.blit(surf, (0, 0))


class VideoApp:
    def __init__(self, config_path="pumpkin.json"):
        # === 設定ファイルの読み込み ===
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # === Pygameの初期化 ===
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("AI_pumpkin_talk_video")
        self.clock = pygame.time.Clock()

        # === 背景動画の読み込み ===
        self.cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
        if not self.cap_bg.isOpened():
            print("背景動画の読み込みに失敗しました。")
            raise RuntimeError("背景動画の読み込みに失敗しました。")
        
        # 背景動画のフレームを保持するための変数
        ret_bg, frame_bg = self.cap_bg.read()
        if not ret_bg:
            self.cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, frame_bg = self.cap_bg.read()
        self.frame_bg = frame_bg
        
        # 背景動画の再生速度制御用 (0.1倍速にするため、10フレームごとに更新)
        self.BG_SPEED_SKIP = BACK_GROUND_SPEED
        self.bg_frame_counter = 0

        # === 各動画の読み込み ===
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
        
        # === 初期化 ===
        self.state = "normal" # 映像の状態 ("normal", "talking", "full2", ...)
        self.dx_main = self.dy_main = self.dx_f3 = self.dy_f3 = self.dx_f6 = self.dy_f6 = 0
        self.current_main = self.current_f3 = self.current_f6 = 0.0
        self.move_start_time_main = self.move_start_time_f3 = self.move_start_time_f6 = 0
        self.start_frame_main = self.start_frame_f3 = self.start_frame_f6 = 0.0
        self.return_to_first_third_main = self.return_to_first_third_f3 = self.return_to_first_third_f6 = False
        
        # === Pygameイベント処理用フラグ ===
        self.running = True
        
        # === 状態変更リクエスト用キュー ===
        self.state_change_queue = queue.Queue()
        
        # === Flaskサーバー用スレッド ===
        self.flask_thread = None

    def change_state(self, new_state):
        """状態を変更する (外部から呼び出し可能)"""
        print(f"[VideoApp] 状態変更リクエスト: {self.state} -> {new_state}")
        # 状態変更リクエストをキューに追加 (スレッドセーフ)
        self.state_change_queue.put(new_state)

    def _process_state_change_requests(self):
        """キューから状態変更リクエストを処理する"""
        try:
            # キューからすべてのリクエストを処理
            while not self.state_change_queue.empty():
                new_state = self.state_change_queue.get_nowait()
                if new_state in ["normal", "talking", "full2", "full3", "full4", "full5", "full6", "full7"]:
                    self.state = new_state
                    print(f"[VideoApp] 状態変更完了: {self.state}")
                else:
                    print(f"[VideoApp] 無効な状態変更要求: {new_state}")
        except queue.Empty:
            pass

    def _update_background(self):
        """背景動画を0.1倍速で更新"""
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
        
        self.bg_frame_counter += 1 # フレームカウンターをインクリメント

    def _draw_current_state(self):
        """現在の状態に応じて描画"""
        t = pygame.time.get_ticks()
        
        # === 各状態描画 ===
        if self.state == "normal":
            dx_main, dy_main = float_motion(t, seed=1, amp_y=22, amp_x=12, base_speed=0.0009)
            draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), dx_main, dy_main, LOWER_GREEN, UPPER_GREEN, self.current_main)
        elif self.state == "talking":
             # 例: talking 状態では、浮遊モーションを少し大きくするなど
             dx_talk, dy_talk = float_motion(t, seed=99, amp_y=25, amp_x=10, base_speed=0.001)
             draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), dx_talk, dy_talk, LOWER_GREEN, UPPER_GREEN, self.current_main)
        elif self.state == "full2":
            ret, frame = self.cap_full2.read()
            if not ret:
                self.cap_full3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.state = "full3"
            else:
                draw_video_fullscreen(self.screen, frame, (SCREEN_W, SCREEN_H), LOWER_GREEN, UPPER_GREEN)
        elif self.state == "full3":
            dx_f3, dy_f3 = float_motion(t, seed=3, amp_y=22, amp_x=12, base_speed=0.0009)
            draw_video(self.screen, self.cap_full3, (SCREEN_W, SCREEN_H), dx_f3, dy_f3, LOWER_GREEN, UPPER_GREEN, self.current_f3)
        elif self.state == "full4":
            ret, frame = self.cap_full4.read()
            if not ret:
                self.current_main = 0
                self.state = "normal"
            else:
                draw_video_fullscreen(self.screen, frame, (SCREEN_W, SCREEN_H), LOWER_GREEN, UPPER_GREEN)
        elif self.state == "full5":
            ret, frame = self.cap_full5.read()
            if not ret:
                self.cap_full6.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.state = "full6"
            else:
                draw_video_fullscreen(self.screen, frame, (SCREEN_W, SCREEN_H), LOWER_GREEN, UPPER_GREEN)
        elif self.state == "full6":
            dx_f6, dy_f6 = float_motion(t, seed=5, amp_y=22, amp_x=12, base_speed=0.0009)
            draw_video(self.screen, self.cap_full6, (SCREEN_W, SCREEN_H), dx_f6, dy_f6, LOWER_GREEN, UPPER_GREEN, self.current_f6)
        elif self.state == "full7":
            ret, frame = self.cap_full7.read()
            if not ret:
                self.current_main = 0
                self.state = "normal"
            else:
                draw_video_fullscreen(self.screen, frame, (SCREEN_W, SCREEN_H), LOWER_GREEN, UPPER_GREEN)
        else: # 未知の状態やデフォルト
             draw_video(self.screen, self.cap_main, (SCREEN_W, SCREEN_H), self.dx_main, self.dy_main, LOWER_GREEN, UPPER_GREEN, self.current_main)

    def run_pygame_loop(self):
        """Pygameのメインループを実行する"""
        print("[VideoApp] Pygameループを開始します...")
        while self.running:
            # Pygameイベントを処理
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.running = False # メインループを停止するフラグを立てる
            
            # 状態変更リクエストを処理
            self._process_state_change_requests()
            
            # 背景動画を更新
            self._update_background()
            
            # 現在の状態に応じて描画
            self._draw_current_state()
            
            # 画面更新
            pygame.display.update()
            self.clock.tick(60) # 60 FPS
        
        # 終了処理
        for cap in [self.cap_main, self.cap_full2, self.cap_full3, self.cap_full4, self.cap_full5, self.cap_full6, self.cap_full7, self.cap_bg]:
            cap.release()
        pygame.quit()
        print("[VideoApp] Pygameループを終了しました。")

    def start_flask_server(self):
        """Flaskサーバーを開始する"""
        app = Flask(__name__)

        @app.route('/signal/change_state', methods=['POST'])
        def change_state_endpoint():
            data = request.get_json()
            new_state = data.get("state", "")
            if new_state:
                self.change_state(new_state)
                return jsonify({"status": "success"}), 200
            else:
                return jsonify({"status": "error", "message": "No state provided"}), 400

        print("[VideoApp] Flaskサーバーを起動します (ポート 5001)...")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

    def run(self):
        """アプリケーションを実行する"""
        print("=== VideoApp を起動します ===")
        
        # Flaskサーバーを別スレッドで開始
        self.flask_thread = threading.Thread(target=self.start_flask_server, daemon=True)
        self.flask_thread.start()
        print("Flaskサーバースレッドを開始しました。")
        
        # Pygameのメインループを実行
        self.run_pygame_loop()
        
        # Flaskサーバースレッドの終了を待つ (最大5秒)
        if self.flask_thread.is_alive():
            self.flask_thread.join(timeout=5)
        print("=== VideoApp を終了します ===")


if __name__ == "__main__":
    video_app = VideoApp("pumpkin.json")
    video_app.run()