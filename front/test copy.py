# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import cv2
import numpy as np
import math
import random

# ==== 設定 ====
BG_VIDEO_PATH = "background.mp4"  # 背景動画
VIDEO_MAIN = "Pumpkin_Center.mp4"
VIDEO_FULL2 = "Pumkin_Center2Left.mp4"
VIDEO_FULL4 = "Pumkin_Left2Center.mp4"
VIDEO_FULL3 = "Pumpkin_Left.mp4"
VIDEO_FULL5 = "Pumkin_Center2Right.mp4"
VIDEO_FULL6 = "Pumkin_Right.mp4"
VIDEO_FULL7 = "Pumkin_Right2Center.mp4"

LOWER_GREEN = np.array([20, 80, 80])
UPPER_GREEN = np.array([105, 255, 255])

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

# ==== 動画描画 ====
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

# ==== メイン ====
def main():
    pygame.init()
    w, h = 1920, 1020
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("AI_pumpkin_talk")
    clock = pygame.time.Clock()

    # === 背景動画の読み込み ===
    cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
    if not cap_bg.isOpened():
        print("背景動画を開けませんでした:", BG_VIDEO_PATH)
        sys.exit()

    bg_fps = cap_bg.get(cv2.CAP_PROP_FPS)
    if bg_fps <= 0:
        bg_fps = 30.0  # 安全値
    bg_speed = 0.2  # ← ここで再生速度(0.5倍速)を設定
    bg_frame_interval = int(1000 / (bg_fps * bg_speed))  # ミリ秒間隔
    last_bg_time = 0
    bg_frame = None

    # 左上画像ロード
    try:
        
        small_image_surface = pygame.transform.scale(small_image_surface, (320, 240))
    except:
        small_image_surface = None

    cap_main = cv2.VideoCapture(VIDEO_MAIN)
    total_main = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))

    state = "normal"
    dx_main = dy_main = 0
    current_main = 0.0
    move_start_time_main = 0
    start_frame_main = 0.0
    return_to_first_third_main = False

    while True:
        t = pygame.time.get_ticks()
        keys = pygame.key.get_pressed()
        a_pressed = keys[K_a]

        # === 背景動画の更新（0.5倍速制御） ===
        if t - last_bg_time >= bg_frame_interval:
            ret_bg, frame = cap_bg.read()
            if not ret_bg:
                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg, frame = cap_bg.read()
            if ret_bg:
                bg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bg_frame = cv2.resize(bg_frame, (w, h))
            last_bg_time = t

        if bg_frame is not None:
            bg_surface = pygame.image.frombuffer(bg_frame.tobytes(), bg_frame.shape[1::-1], "RGB")
            screen.blit(bg_surface, (0, 0))

        # イベント処理
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                for cap in [cap_bg, cap_main]:
                    cap.release()
                pygame.quit()
                sys.exit()

        # 左上画像
        if small_image_surface:
            sx, sy = float_motion(t, seed=99, amp_x=20, amp_y=20)
            screen.blit(small_image_surface, (10 + sx, 10 + sy))

        # === Aキー処理 ===
        def handle_a_key(cap, total, current, start_frame, move_start_time, return_to_first_third, seed):
            dx, dy = float_motion(t, seed=seed, amp_y=22, amp_x=12, base_speed=0.0009)
            duration = 230
            if a_pressed:
                progress = (t - move_start_time) / duration
                if return_to_first_third:
                    first_third_frame = random.randint(0, max(1, total // 3))
                    if progress >= 1.0:
                        current = first_third_frame
                        return_to_first_third = False
                        move_start_time = t
                        start_frame = current
                    else:
                        ease = ease_in_out_sine(min(progress, 1))
                        current = start_frame + (first_third_frame - start_frame) * ease
                else:
                    random_frame = random.randint(total // 2, total - 1)
                    if progress >= 1.0:
                        current = random_frame
                        return_to_first_third = True
                        move_start_time = t
                        start_frame = current
                    else:
                        if move_start_time == 0:
                            move_start_time = t
                            start_frame = current
                        ease = ease_in_out_sine(min(progress, 1))
                        current = start_frame + (random_frame - start_frame) * ease
            else:
                move_start_time = 0
                current = 0
            return dx, dy, current, start_frame, move_start_time, return_to_first_third

        # メイン状態
        if state == "normal":
            dx_main, dy_main, current_main, start_frame_main, move_start_time_main, return_to_first_third_main = \
                handle_a_key(cap_main, total_main, current_main, start_frame_main, move_start_time_main, return_to_first_third_main, seed=1)
            draw_video(screen, cap_main, (w, h), dx_main, dy_main, LOWER_GREEN, UPPER_GREEN, current_main)

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()
