import pygame
import cv2
import sys

# 動画ファイルを数字キーに割り当て
video_map = {
    pygame.K_1: "Pumpkin-Entry.mov",
    pygame.K_2: "Pumpkin-Finish.mov",
    pygame.K_3: "video3.mov"
}

def play_video(screen, filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"動画を開けません: {filename}")
        return

    clock = pygame.time.Clock()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCVはBGRなのでRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.transpose(frame)  # 90度回転が必要な場合は調整
        frame_surface = pygame.surfarray.make_surface(frame)

        # 画面に描画
        screen.blit(pygame.transform.scale(frame_surface, screen.get_size()), (0, 0))
        pygame.display.flip()

        # FPS制御
        clock.tick(30)

        # イベント処理（Escで中断）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                cap.release()
                return

    cap.release()

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("MOV Player with Number Keys")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in video_map:
                    play_video(screen, video_map[event.key])

        screen.fill((0, 0, 0))
        pygame.display.flip()

if __name__ == "__main__":
    main()
