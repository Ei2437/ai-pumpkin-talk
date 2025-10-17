# -*- coding: utf-8 -*-
import pygame
from pygame.locals import *
import sys

def main():
    pygame.init() # 初期化
    (w, h) = (1000, 650) #画面のサイズ
    (x, y) = (w/2, h/2)
    pygame.display.set_mode((w, h), 0, 32)
    screen = pygame.display.get_surface()
    pygame.display.set_caption("AI_pumpkin_talk") # ウィンドウの上の方に出てくるアレの指定
    bg = pygame.image.load("7725950.jpg").convert_alpha() # 背景画像の指定
    rect_bg = bg.get_rect() # 画像のサイズ取得？？だと思われる
    player = pygame.image.load("panpkin.png").convert_alpha() # キャラ画像の指定
    rect_player = player.get_rect() # 謎 上のキャラ画像の何かを取得してるっぽい
    rect_player.center = (x, y) # キャラ座標

    while(True):
        screen.fill((0, 0, 0, 0)) # 背景色の指定。RGBのはず
        screen.blit(bg, rect_bg) # 背景画像の描画
        screen.blit(player, rect_player) # キャラの描画
        pygame.time.wait(10) # 更新間隔。多分ミリ秒
        pygame.display.update() # 画面更新
        x += 1 # 画面更新毎にx座標を+1
        y += 1 # 画面更新毎にy座標を+1
        rect_player.center = (x, y)
        if x > w: # 端まで来たら座標を0にリセット
            x = 0
        if y > h: # 同上
            y = 0

        for event in pygame.event.get(): # 終了処理
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

if __name__ == "__main__":
    main()