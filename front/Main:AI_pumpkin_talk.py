# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import cv2
import numpy as np
import math
import random
import time

# ==== 設定 ====
BG_IMAGE_PATH = "1000000227.png"
VIDEO_MAIN = "Pumpkin-Center.mp4"
VIDEO_FULL2 = "Pumpkin-Center2Left.mp4"
VIDEO_FULL3 = "Pumpkin-Left.mp4"
VIDEO_FULL4 = "Pumpkin-Left2Center.mp4"
VIDEO_SMALL = "93836-641767488_small.mp4"

# オレンジ透過閾値 (HSV)
LOWER_ORANGE = np.array([5, 80, 80])
UPPER_ORANGE = np.array([30, 255, 255])

# グリーンバック透過閾値 (HSV)
LOWER_GREEN = np.array([35, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])


def chroma_key_rgba(frame_rgb, lower, upper):
    """指定色を透過してRGBAに変換"""
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.medianBlur(mask, 3)
    alpha = 255 - mask
    rgba = np.dstack((frame_rgb, alpha))
    return rgba


# ==== 浮遊モーション ====
def float_motion(t, seed=0, amp_y=16, amp_x=7, base_speed=0.0007):
    """ふわふわ浮く動き"""
    random.seed(seed)
    dy = math.sin(t * base_speed + seed) * amp_y
    dx = (
        math.sin(t * base_speed * 1.2 + seed * 2.3) * amp_x
        + math.cos(t * base_speed * 0.7 + seed * 1.5) * amp_x * 0.6
        + math.sin(t * base_speed * 0.25 + seed * 4.7) * amp_x * 0.3
    )
    return int(dx), int(dy)


# ==== アニメーション的に中央へ戻す ====
def animate_to_center(current_offset, duration=400):
    """dx,dyをゆっくり(400ms)で0に戻す"""
    start_time = pygame.time.get_ticks()
    start_dx, start_dy = current_offset
    while True:
        elapsed = pygame.time.get_ticks() - start_time
        if elapsed > duration:
            break
        progress = elapsed / duration
        ease = 0.5 - 0.5 * math.cos(math.pi * progress)  # イージング
        dx = int(start_dx * (1 - ease))
        dy = int(start_dy * (1 - ease))
        yield dx, dy
    yield 0, 0


def main():
    pygame.init()
    (w, h) = (1900, 1000)
    screen = pygame.display.set_mode((w, h), 0, 32)
    pygame.display.set_caption("AI_pumpkin_talk")
    clock = pygame.time.Clock()

    # 背景画像
    try:
        bg_image = pygame.image.load(BG_IMAGE_PATH).convert()
        bg_image = pygame.transform.scale(bg_image, (w, h))
    except FileNotFoundError:
        bg_image = None

    # ==== 各動画 ====
    cap_main = cv2.VideoCapture(VIDEO_MAIN)
    cap_full2 = cv2.VideoCapture(VIDEO_FULL2)
    cap_full3 = cv2.VideoCapture(VIDEO_FULL3)
    cap_full4 = cv2.VideoCapture(VIDEO_FULL4)
    cap_small = cv2.VideoCapture(VIDEO_SMALL)

    total_frames_main = int(cap_main.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_full3 = int(cap_full3.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame_main = 0.0
    current_frame_full3 = 0.0
    prev_mouse_y = h // 2
    speed_factor = 0.5
    small_w, small_h = 320, 240

    state = "normal"
    dx_main = dy_main = dx_f3 = dy_f3 = 0

    while True:
        t = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                for cap in [cap_main, cap_full2, cap_full3, cap_full4, cap_small]:
                    cap.release()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # === 状態ごとの切り替え処理 ===
                    if state == "normal":
                        # ふわふわ位置を中央へ戻すアニメーション
                        for dx_main, dy_main in animate_to_center((dx_main, dy_main)):
                            draw_frame(screen, bg_image, cap_small, small_w, small_h)
                            draw_video_center(
                                screen,
                                cap_main,
                                (w, h),
                                dx_main,
                                dy_main,
                                LOWER_GREEN,
                                UPPER_GREEN,
                                current_frame_main,
                            )
                            pygame.display.update()
                            clock.tick(60)
                        cap_full2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        state = "full2"

                    elif state == "full3":
                        for dx_f3, dy_f3 in animate_to_center((dx_f3, dy_f3)):
                            draw_frame(screen, bg_image, cap_small, small_w, small_h)
                            draw_video_center(
                                screen,
                                cap_full3,
                                (w, h),
                                dx_f3,
                                dy_f3,
                                LOWER_GREEN,
                                UPPER_GREEN,
                                current_frame_full3,
                            )
                            pygame.display.update()
                            clock.tick(60)
                        cap_full4.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        state = "full4"

        # ==== 背景 ====
        draw_frame(screen, bg_image, cap_small, small_w, small_h)

        # ==== 状態別処理 ====
        if state == "full2":
            ret, frame = cap_full2.read()
            if not ret:
                cap_full3.set(cv2.CAP_PROP_POS_FRAMES, 0)
                state = "full3"
            else:
                draw_video_full(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)

        elif state == "full3":
            dx_f3, dy_f3 = float_motion(t, seed=3, amp_y=22, amp_x=12, base_speed=0.0009)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dy = prev_mouse_y - mouse_y
            prev_mouse_y = mouse_y
            current_frame_full3 += dy * speed_factor
            current_frame_full3 = max(0.0, min(total_frames_full3 - 1, current_frame_full3))
            cap_full3.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_full3))
            ret, frame = cap_full3.read()
            if not ret:
                state = "full4"
                cap_full4.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                draw_video_center(screen, cap_full3, (w, h), dx_f3, dy_f3, LOWER_GREEN, UPPER_GREEN, current_frame_full3)

        elif state == "full4":
            ret, frame = cap_full4.read()
            if not ret:
                current_frame_main = 0.0
                state = "normal"
            else:
                draw_video_full(screen, frame, (w, h), LOWER_GREEN, UPPER_GREEN)

        else:
            dx_main, dy_main = float_motion(t, seed=1, amp_y=22, amp_x=12, base_speed=0.0009)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dy = prev_mouse_y - mouse_y
            prev_mouse_y = mouse_y
            current_frame_main += dy * speed_factor
            current_frame_main = max(0.0, min(total_frames_main - 1, current_frame_main))
            cap_main.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame_main))
            ret, frame_main = cap_main.read()
            if ret:
                draw_video_center(screen, cap_main, (w, h), dx_main, dy_main, LOWER_GREEN, UPPER_GREEN, current_frame_main)

        pygame.display.update()
        clock.tick(60)


# ==== 共通描画関数 ====
def draw_frame(screen, bg_image, cap_small, small_w, small_h):
    if bg_image:
        screen.blit(bg_image, (0, 0))
    else:
        screen.fill((0, 0, 0))
    if cap_small.isOpened():
        ret_small, frame_small = cap_small.read()
        if ret_small:
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            frame_small = cv2.resize(frame_small, (small_w, small_h))
            rgba_small = chroma_key_rgba(frame_small, LOWER_ORANGE, UPPER_ORANGE)
            surf_small = pygame.image.frombuffer(rgba_small.tobytes(), rgba_small.shape[1::-1], "RGBA")
            surf_small = surf_small.convert_alpha()
            screen.blit(surf_small, (0, 0))


def draw_video_full(screen, frame, size, lower, upper):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, size)
    rgba = chroma_key_rgba(frame, lower, upper)
    surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
    surf = surf.convert_alpha()
    screen.blit(surf, (0, 0))


def draw_video_center(screen, cap, size, dx, dy, lower, upper, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        rgba = chroma_key_rgba(frame, lower, upper)
        surf = pygame.image.frombuffer(rgba.tobytes(), rgba.shape[1::-1], "RGBA")
        surf = surf.convert_alpha()
        screen.blit(surf, (dx, dy))


if __name__ == "__main__":
    main()
