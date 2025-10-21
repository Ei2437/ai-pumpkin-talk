# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys
import cv2
import numpy as np
import math
import random
from flask import Flask, request, jsonify # Flask追加

# ==== 設定 ====
# ---- 動画 ----
BG_VIDEO_PATH = "background.mp4"
VIDEO_MAIN = "Pumpkin_Center.mp.mp4"
VIDEO_FULL2 = "Pumkin_Center2Left.mp.mp4"
VIDEO_FULL4 = "Pumkin_Left2Center.mp.mp4"
VIDEO_FULL3 = "Pumpkin_Left.mp.mp4"
VIDEO_FULL5 = "Pumkin_Center2Right.mp.mp4"
VIDEO_FULL6 = "Pumkin_Right.mp.mp4"
VIDEO_FULL7 = "Pumkin_Right2Center.mp.mp4"

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

# Flaskアプリケーションの初期化
app = Flask(__name__)

# Aキーイベントのカウンター (アニメーション状態を更新するためのトリガー)
a_key_event_count = 0
# 各キャプチャごとのアニメーション状態を保持する辞書
animation_state = {
    "main": {"dx": 0, "dy": 0, "current": 0.0, "start_frame": 0.0, "move_start_time": 0, "return_to_first_third": False},
    "full3": {"dx": 0, "dy": 0, "current": 0.0, "start_frame": 0.0, "move_start_time": 0, "return_to_first_third": False},
    "full6": {"dx": 0, "dy": 0, "current": 0.0, "start_frame": 0.0, "move_start_time": 0, "return_to_first_third": False}
}

@app.route('/key_event', methods=['POST'])
def receive_key_event():
    global a_key_event_count
    data = request.get_json()
    key_name = data.get("key")

    if key_name == 'left':
        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_LEFT))
        print("LEFT key event posted to pygame queue")
    elif key_name == 'right':
        pygame.event.post(pygame.event.Event(KEYDOWN, key=K_RIGHT))
        print("RIGHT key event posted to pygame queue")
    elif key_name == 'a':
        # Aキーが押されたイベントを受信 -> カウンターをインクリメント
        a_key_event_count += 1
        print("A key event received (count incremented)")
    else:
        print(f"Unknown key received: {key_name}")
        return jsonify({"status": "error", "message": "Unknown key"}), 400

    return jsonify({"status": "success"})

# ==== メイン ====
def main():
    global a_key_event_count, animation_state
    pygame.init()
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("AI_pumpkin_talk")
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
    # dx_main, dy_main, dx_f3, dy_f3, dx_f6, dy_f6 = 0 # 削除
    # current_main = current_f3 = current_f6 = 0.0 # 削除
    # move_start_time_main = move_start_time_f3 = move_start_time_f6 = 0 # 削除
    # start_frame_main = start_frame_f3 = start_frame_f6 = 0.0 # 削除
    # return_to_first_third_main = return_to_first_third_f3 = return_to_first_third_f6 = False # 削除

    # Flaskサーバーを別スレッドで起動
    import threading
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False))
    server_thread.daemon = True
    server_thread.start()

    # Aキーイベント監視用の前回カウント
    prev_a_key_count = 0

    while True:
        t = pygame.time.get_ticks()
        # keys = pygame.key.get_pressed() # 使用しない
        # a_pressed = keys[K_a] # 使用しない

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

        # --- Aキーイベントによるフレーム移動処理 ---
        # Aキーイベントが発生した回数が前回から増えたら、handle_a_keyを1回実行
        current_a_count = a_key_event_count
        if current_a_count > prev_a_key_count:
            # 1回だけ実行
            if state == "normal":
                animation_state["main"] = handle_a_key(cap_main, total_main, animation_state["main"], seed=1, t=t)
            elif state == "full3":
                animation_state["full3"] = handle_a_key(cap_full3, total_f3, animation_state["full3"], seed=3, t=t)
            elif state == "full6":
                animation_state["full6"] = handle_a_key(cap_full6, total_f6, animation_state["full6"], seed=5, t=t)
            # 他の状態では実行しない
            prev_a_key_count = current_a_count # カウントを更新

        # === 各状態描画 ===
        if state == "normal":
            dx_main, dy_main = animation_state["main"]["dx"], animation_state["main"]["dy"]
            current_main = animation_state["main"]["current"]
            draw_video(screen, cap_main, (w, h), dx_main, dy_main, LOWER_GREEN, UPPER_GREEN, current_main)
        elif state == "full3":
            dx_f3, dy_f3 = animation_state["full3"]["dx"], animation_state["full3"]["dy"]
            current_f3 = animation_state["full3"]["current"]
            draw_video(screen, cap_full3, (w, h), dx_f3, dy_f3, LOWER_GREEN, UPPER_GREEN, current_f3)
        elif state == "full6":
            dx_f6, dy_f6 = animation_state["full6"]["dx"], animation_state["full6"]["dy"]
            current_f6 = animation_state["full6"]["current"]
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

def handle_a_key(cap, total, state_dict, seed, t):
    """Aキーイベント1回分のアニメーション状態更新"""
    dx, dy = float_motion(t, seed=seed, amp_y=22, amp_x=12, base_speed=0.0009)
    duration = 230
    # 1回のイベントで、アニメーションを1ステップ進める
    # 既存のロジックを流用するため、progressを1フレーム分進めると仮定
    # ただし、イベントベースなので、move_start_time をイベント発生時刻に更新
    # そして、ease_in_out_sine を使用して補間を進める
    # シード値ごとの状態をstate_dictで管理
    current = state_dict["current"]
    start_frame = state_dict["start_frame"]
    move_start_time = state_dict["move_start_time"]
    return_to_first_third = state_dict["return_to_first_third"]

    # move_start_time が0の場合、初期状態とみなす
    if move_start_time == 0:
        move_start_time = t
        start_frame = current
        # 最初はランダムフレームに移動
        if not return_to_first_third:
            target_frame = random.randint(total // 2, total - 1)
        else:
            target_frame = random.randint(0, max(1, total // 3))
        state_dict["start_frame"] = start_frame
        state_dict["current"] = target_frame
        state_dict["move_start_time"] = move_start_time
        state_dict["return_to_first_third"] = not return_to_first_third
        # dx, dy は float_motion で計算
        state_dict["dx"] = dx
        state_dict["dy"] = dy
        return state_dict

    # 前回のイベントからの経過時間で補間
    progress = (t - move_start_time) / duration
    if progress >= 1.0:
        # 前のターゲットに到達
        if return_to_first_third:
            new_target = random.randint(0, max(1, total // 3))
        else:
            new_target = random.randint(total // 2, total - 1)
        # 次の移動のための準備
        state_dict["start_frame"] = current
        state_dict["current"] = new_target
        state_dict["move_start_time"] = t # 新しいイベント時刻
        state_dict["return_to_first_third"] = not return_to_first_third
    else:
        # 補間中
        ease = ease_in_out_sine(min(progress, 1))
        state_dict["current"] = start_frame + (current - start_frame) * ease

    # dx, dy は float_motion で計算
    state_dict["dx"] = dx
    state_dict["dy"] = dy
    return state_dict


if __name__ == "__main__":
    main()