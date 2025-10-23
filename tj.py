#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テキスト音声合成ファイル書き出しプログラム
入力されたテキストをVOICEVOXで音声合成し、WAVファイルとして保存します。
"""

import os
import json
import requests
import numpy as np
from scipy.io import wavfile
import argparse


class TextToSpeechWriter:
    def __init__(self, config_path="pumpkin.json"):
        """設定ファイルを読み込んで初期化"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        # VOICEVOX設定の読み込み
        voicevox_config = self.config.get("system", {}).get("voicevox", {})
        self.voicevox_url = voicevox_config.get("url", "http://localhost:50021")
        self.speaker_id = voicevox_config.get("speaker_id", 1)
        self.speed = voicevox_config.get("speed", 1.0)
        self.pitch = voicevox_config.get("pitch", 0.0)
        self.intonation = voicevox_config.get("intonation", 1.0)
        self.volume = voicevox_config.get("volume", 1.0)
        self.post_phoneme_length = voicevox_config.get("post_phoneme_length", 0.1)
    
    def synthesize_speech(self, text):
        """テキストから音声データを生成"""
        try:
            # 1. 音声クエリの作成
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            # 2. パラメータの適用
            query_data["speedScale"] = self.speed
            query_data["pitchScale"] = self.pitch
            query_data["intonationScale"] = self.intonation
            query_data["volumeScale"] = self.volume
            query_data["postPhonemeLength"] = self.post_phoneme_length
            
            # 3. 音声合成
            synthesis_url = f"{self.voicevox_url}/synthesis"
            synthesis_params = {"speaker": self.speaker_id}
            synthesis_response = requests.post(
                synthesis_url,
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            synthesis_response.raise_for_status()
            
            return synthesis_response.content
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None
    
    def save_audio(self, audio_data, output_path):
        """音声データをWAVファイルとして保存"""
        if audio_data is None:
            print("保存する音声データがありません")
            return False
        
        try:
            # WAVファイルとして保存
            with open(output_path, "wb") as f:
                f.write(audio_data)
            print(f"音声ファイルを保存しました: {output_path}")
            return True
            
        except Exception as e:
            print(f"ファイル保存中にエラーが発生しました: {e}")
            return False
    
    def text_to_file(self, text, output_path):
        """テキストを音声合成してファイルに保存（メイン処理）"""
        print(f"音声合成中: {text}")
        audio_data = self.synthesize_speech(text)
        
        if audio_data:
            return self.save_audio(audio_data, output_path)
        return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="テキストを音声合成してWAVファイルとして保存します"
    )
    parser.add_argument(
        "text",
        nargs='+',
        help="音声合成するテキスト"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="出力するWAVファイル名（例: output.wav）"
    )
    parser.add_argument(
        "-c", "--config",
        default="pumpkin.json",
        help="設定ファイルのパス（デフォルト: pumpkin.json）"
    )
    
    args = parser.parse_args()
    
    # テキストを結合
    text = ' '.join(args.text)
    
    # 出力ファイル名に拡張子がない場合は.wavを追加
    output_path = args.output
    if not output_path.lower().endswith('.wav'):
        output_path += '.wav'
    
    try:
        # 音声合成・保存の実行
        tts_writer = TextToSpeechWriter(args.config)
        success = tts_writer.text_to_file(args.text, output_path)
        
        if success:
            print("処理が完了しました")
            return 0
        else:
            print("処理に失敗しました")
            return 1
            
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return 1
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

# python tj.py {text} -o output.wav