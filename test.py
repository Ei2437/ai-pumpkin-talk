# ver2.1  10/11 13:00 (video_app.py にシグナル送信版)

import os
import io
import time
import json
import re
import random
import requests
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from scipy.io import wavfile
from flask import Flask, request, jsonify
import threading

# ==== プロキシ完全無効化 ====
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

REQUESTS_NO_PROXY = {"http": None, "https": None}
# =============================


class LoadConfig:
    def __init__(self, config_path="pumpkin.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def get_character_prompt(self):
        char = self.config["character"]
        # 常に character.prompt を使用
        knowledge_dict = self.config.get("knowledge", {})
        knowledge_str = ""
        for category, items in knowledge_dict.items():
            knowledge_str += f"\n【{category}】\n" + "\n".join(items) + "\n"
        
        return char["prompt"].format(knowledge=knowledge_str)
    
    def get_ollama_config(self):
        return self.config["api"]["ollama"]
    
    def get_voicevox_config(self):
        return self.config["api"]["voicevox"]
    
    def get_system_config(self):
        return self.config.get("system", {})
    
    def get_advanced_config(self):
        return self.config.get("advanced", {})


class PumpkinTalk:
    def __init__(self, config_path="pumpkin.json"):
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

        # === 映像アプリとの連携 ===
        self.video_app_url = "http://localhost:5001" # video_app.py のFlaskサーバーURL

    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        try:
            self.conversation_history.append(f"ユーザー: {input_text}")
            recent_history = "\n".join(self.conversation_history[-6:])
            
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": False,
                "options": self.ollama_config.get("params", {})
            }
            
            response = requests.post(url, json=payload, proxies=REQUESTS_NO_PROXY)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "応答を生成できませんでした。")
            
            # advanced 設定が存在する場合のみフィルタリングを実行
            if self.advanced_config:
                response_text = self.filter_response(response_text)
            
            self.conversation_history.append(f"パンプキン: {response_text}")
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぜ！もう一度話しかけてみろよ！"

    def text_to_speech(self, text):
        try:
            # 1. テキストから音声合成用のクエリを作成
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params, proxies=REQUESTS_NO_PROXY)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            # 2. jsonの適用
            if "voicevox" in self.system_config:
                voicevox_settings = self.system_config["voicevox"]
                # 速度調整
                if "speed" in voicevox_settings:
                    query_data["speedScale"] = voicevox_settings["speed"]
                # 音程調整
                if "pitch" in voicevox_settings:
                    query_data["pitchScale"] = voicevox_settings["pitch"]
                # 抑揚調整
                if "intonation" in voicevox_settings:
                    query_data["intonationScale"] = voicevox_settings["intonation"]
                # 音量調整
                if "volume" in voicevox_settings:
                    query_data["volumeScale"] = voicevox_settings["volume"]
                # 音素後の余白
                if "post_phoneme_length" in voicevox_settings:
                    query_data["postPhonemeLength"] = voicevox_settings["post_phoneme_length"]
            
            # 3. 音声合成
            synthesis_url = f"{self.voicevox_url}/synthesis"
            synthesis_params = {"speaker": self.speaker_id}
            synthesis_response = requests.post(
                synthesis_url, 
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"},
                proxies=REQUESTS_NO_PROXY
            )
            synthesis_response.raise_for_status()
            
            # 音声データを取得
            wav_data = io.BytesIO(synthesis_response.content)
            
            # WAVデータの読み込み
            wav_data.seek(0)
            sample_rate, audio_data = wavfile.read(wav_data)
            
            # モノラルならステレオに
            if len(audio_data.shape) == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None

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

    def send_signal_to_video_app(self, signal_type):
        """video_app.py にシグナルを送信する"""
        url = f"{self.video_app_url}/signal/{signal_type}"
        try:
            response = requests.post(url, json={}, proxies=REQUESTS_NO_PROXY)
            response.raise_for_status()
            print(f"[pc1] 映像アプリにシグナル '{signal_type}' を送信しました。")
        except requests.exceptions.RequestException as e:
            print(f"[pc1] 映像アプリへのシグナル送信に失敗しました ({url}): {e}")

    def process_input_text(self, input_text):
        """音声応答と映像状態の連携処理"""
        
        # 1. 映像を "talking" 状態に変更 (例: full2を開始)
        self.send_signal_to_video_app("start_full2")

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
        self.send_signal_to_video_app("reset_to_normal")
        print("処理完了。")

    def filter_response(self, response_text):
        """応答テキストをフィルタリング (advanced 設定が存在する場合のみ呼び出される)"""
        # advanced 設定が存在する場合のみフィルタリングを実行
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            # 除去
            if "remove_patterns" in filtering:
                import re
                for pattern in filtering["remove_patterns"]:
                    response_text = re.sub(pattern, "", response_text)
            
            # 置換
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()

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
    print("=== PumpkinTalk 統合版 (音声 + 映像シグナル送信) を起動します ===")
    
    # 1. PumpkinTalk インスタンスを作成
    try:
        pumpkin_talk = PumpkinTalk("pumpkin.json")
    except Exception as e:
        print(f"PumpkinTalk の初期化に失敗しました: {e}")
        return

    # 2. Flask サーバーを起動
    app = create_flask_app(pumpkin_talk)
    print("Flaskサーバーを起動します...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False) # use_reloader=False でスレッドの問題を回避

if __name__ == "__main__":
    main()