# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import av
import cv2
import numpy as np
import math
import random
from flask import Flask, request, jsonify # Flask追加

# ==== 設定 ====
w, h = 1920, 1020
BG_VIDEO_PATH = "BG.mp4"

VIDEO_MAIN = "Pumpkin-Center.mov"
VIDEO_FULL2 = "Pumpkin-Center2Left.mov"
VIDEO_FULL3 = "Pumpkin-Left.mov"
VIDEO_FULL4 = "Pumpkin-Left2Center.mov"
VIDEO_FULL5 = "Pumpkin-Center2Right.mov"
VIDEO_FULL6 = "Pumpkin-Right.mov"
VIDEO_FULL7 = "Pumpkin-Right2Center.mov"

BACK_SPEED_SKIP = 10  # 背景動画速度（フレーム単位）

# Flaskアプリケーションの初期化
app = Flask(__name__)

# Aキーのトグル状態を管理するグローバル変数
a_key_active = False

# ==== 浮遊モーション ====
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
        self.direction = 1  # 1: 順方向, -1: 逆方向

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

    if a_key_active:  # トグルがONの場合
        # フレームを進める（方向に応じて）
        vid.current_frame += vid.direction
        
        # 最後まで行ったら逆方向に
        if vid.current_frame >= vid.total - 1:
            vid.current_frame = vid.total - 1
            vid.direction = -1
        # 最初まで戻ったら順方向に
        elif vid.current_frame <= 0:
            vid.current_frame = 0
            vid.direction = 1
    else:  # トグルがOFFの場合
        vid.current_frame = 0
        vid.direction = 1

    return vid.get_frame(), dx, dy

# Flaskエンドポイント
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
    global state, a_key_active
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
    import threading
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False))
    server_thread.daemon = True
    server_thread.start()

    while True:
        t = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == QUIT or (event.type==KEYDOWN and event.key==K_ESCAPE):
                pygame.quit()
                sys.exit()
            # Aキーでトグル
            elif event.type == KEYDOWN and event.key == K_a:
                a_key_active = not a_key_active
                print(f"A key toggled: {'ON' if a_key_active else 'OFF'}")
            # LEFT/RIGHTキー入力処理
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

        # 背景描画（スロー再生）
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