# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import cv2
import numpy as np
import math
import random

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
w = 1920 #横幅
h = 1020 #高さ

#---- 背景動画の再生速度 ----
back_ground_speed  = 10 #フレームで管理。10なら0.1倍速になる


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

# ==== メイン ====
def main():
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("AI_pumpkin_talk") #画面上部の名前
    clock = pygame.time.Clock()

    # 背景動画を読み込み
    cap_bg = cv2.VideoCapture(BG_VIDEO_PATH)
    if not cap_bg.isOpened():
        print("背景動画の読み込みに失敗しました。")
        sys.exit()
    
    # 背景動画のフレームを保持するための変数
    # 初期フレームを読み込んでおく
    ret_bg, frame_bg = cap_bg.read()
    if not ret_bg:
        cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, frame_bg = cap_bg.read()
    
    # 背景動画の再生速度制御用 (0.1倍速にするため、10フレームごとに更新)
    BG_SPEED_SKIP = back_ground_speed
    bg_frame_counter = 0

    cap_main = cv2.VideoCapture(VIDEO_MAIN)
    cap_full2 = cv2.VideoCapture(VIDEO_FULL2)
    cap_full3 = cv2.VideoCapture(VIDEO_FULL3)
    cap_full4 = cv2.VideoCapture(VIDEO_FULL4)
    cap_full5 = cv2.VideoCapture(VIDEO_FULL5)
    cap_full6 = cv2.VideoCapture(VIDEO_FULL6)
    cap_full7 = cv2.VideoCapture(VIDEO_FULL7)

    total_main = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))
    total_f3 = int(cap_full3.get(cv2.CAP_PROP_FRAME_COUNT))
    total_f6 = int(cap_full6.get(cv2.CAP_PROP_FRAME_COUNT))
    #初期化
    state = "normal"
    dx_main = dy_main = dx_f3 = dy_f3 = dx_f6 = dy_f6 = 0
    current_main = current_f3 = current_f6 = 0.0
    move_start_time_main = move_start_time_f3 = move_start_time_f6 = 0
    start_frame_main = start_frame_f3 = start_frame_f6 = 0.0
    return_to_first_third_main = return_to_first_third_f3 = return_to_first_third_f6 = False

    while True:
        t = pygame.time.get_ticks()
        keys = pygame.key.get_pressed()
        a_pressed = keys[K_a]

        for event in pygame.event.get():
            # 画面を閉じる条件（右上の罰を押すかEscキーを押すと閉じる）
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                for cap in [cap_main, cap_full2, cap_full3, cap_full4, cap_full5, cap_full6, cap_full7, cap_bg]:
                    cap.release()
                pygame.quit()
                sys.exit()
            #左矢印キーが押された時の処理
            elif event.type == KEYDOWN and event.key == K_LEFT:
                #ぱんぷきんが真ん中にいた時の処理（ぱんぷきんが左へ移動する）
                if state == "normal":
                    cap_full2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state = "full2"
                #ぱんぷきんが左にいた時の処理（ぱんぷきんが真ん中へ移動する）
                elif state == "full3":
                    cap_full4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state = "full4"
            #右矢印キーが押された時の処理
            elif event.type == KEYDOWN and event.key == K_RIGHT:
                #ぱんぷきんが真ん中にいた時の処理（ぱんぷきんが右へ移動する）
                if state == "normal":
                    cap_full5.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state = "full5"
                #ぱんぷきんが右にいたの処理（ぱんぷきんが真ん中へ移動する）
                elif state == "full6":
                    cap_full7.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    state = "full7"
        # ==== 背景動画を0.1倍速で更新 ====
        if bg_frame_counter % BG_SPEED_SKIP == 0:
            ret_bg_new, frame_bg_new = cap_bg.read()
            if not ret_bg_new:
                cap_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg_new, frame_bg_new = cap_bg.read()
            frame_bg = frame_bg_new # 新しいフレームを保持
        
        bg_rgb = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)
        bg_rgb = cv2.resize(bg_rgb, (w, h))
        bg_surface = pygame.image.frombuffer(bg_rgb.tobytes(), bg_rgb.shape[1::-1], "RGB")
        screen.blit(bg_surface, (0, 0))
        
        bg_frame_counter += 1 # フレームカウンターをインクリメント

        # --- Aキー連続フレーム移動処理 ---
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

        # === 各状態描画 ===
        if state == "normal":
            dx_main, dy_main, current_main, start_frame_main, move_start_time_main, return_to_first_third_main = \
                handle_a_key(cap_main, total_main, current_main, start_frame_main, move_start_time_main, return_to_first_third_main, seed=1)
            draw_video(screen, cap_main, (w, h), dx_main, dy_main, LOWER_GREEN, UPPER_GREEN, current_main)
        elif state == "full3":
            dx_f3, dy_f3, current_f3, start_frame_f3, move_start_time_f3, return_to_first_third_f3 = \
                handle_a_key(cap_full3, total_f3, current_f3, start_frame_f3, move_start_time_f3, return_to_first_third_f3, seed=3)
            draw_video(screen, cap_full3, (w, h), dx_f3, dy_f3, LOWER_GREEN, UPPER_GREEN, current_f3)
        elif state == "full6":
            dx_f6, dy_f6, current_f6, start_frame_f6, move_start_time_f6, return_to_first_third_f6 = \
                handle_a_key(cap_full6, total_f6, current_f6, start_frame_f6, move_start_time_f6, return_to_first_third_f6, seed=5)
            draw_video(screen, cap_full6, (w, h), dx_f6, dy_f6, LOWER_GREEN, UPPER_GREEN, current_f6)
        elif state == "full2":
            ret, frame = cap_full2.read()
            if not ret:
                cap_full3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = "full3"
            else:
                draw_video_fullscreen(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)
        elif state == "full4":
            ret, frame = cap_full4.read()
            if not ret:
                current_main = 0
                state = "normal"
            else:
                draw_video_fullscreen(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)
        elif state == "full5":
            ret, frame = cap_full5.read()
            if not ret:
                cap_full6.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = "full6"
            else:
                draw_video_fullscreen(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)
        elif state == "full7":
            ret, frame = cap_full7.read()
            if not ret:
                current_main = 0
                state = "normal"
            else:
                draw_video_fullscreen(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()