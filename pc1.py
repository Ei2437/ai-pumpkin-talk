# ver1.9  10/10 23:30

import os
import time
import json
import requests
import numpy as np
from scipy.io import wavfile
from flask import Flask, request, jsonify
import subprocess

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
            
            response = requests.post(url, json=payload)
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
            query_response = requests.post(query_url, params=query_params)
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
                headers={"Content-Type": "application/json"}
            )
            synthesis_response.raise_for_status()
            
            # 4. 音声データを取得
            wav_data = synthesis_response.content
            
            # 5. WAVファイルに書き出し (上書き)
            with open(self.temp_wav_file, "wb") as f:
                f.write(wav_data)
            print(f"音声ファイルを '{self.temp_wav_file}' に書き出しました")
            
            # 6. wavfile.read でサンプリングレートとデータを取得 (確認用)
            sample_rate, audio_data = wavfile.read(self.temp_wav_file)
            
            # 7. モノラルならステレオに
            if len(audio_data.shape) == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None

    def play_audio_with_aplay(self):
        """aplay を使用して音声ファイルを再生"""
        if not os.path.exists(self.temp_wav_file):
            print("再生できる音声ファイルがありません")
            return
        
        # ファイルサイズを確認
        if os.path.getsize(self.temp_wav_file) == 0:
            print("警告: 音声ファイルが空です")
            return

        try:
            print(f"'{self.temp_wav_file}' を aplay で再生中...")
            # -q オプションで再生のみ (メッセージを抑制)
            result = subprocess.run(["aplay", "-q", self.temp_wav_file])
            if result.returncode == 0:
                print("再生完了")
            else:
                print(f"aplay でエラーが発生しました (終了コード: {result.returncode})")
        except FileNotFoundError:
            print("aplay が見つかりません。'sudo apt install alsa-utils' でインストールしてください。")
        except Exception as e:
            print(f"音声再生中にエラーが発生しました: {e}")

    def process_input_text(self, input_text):
        # 応答の生成
        response_text = self.generate_response(input_text)
        print("回答:", response_text)
        
        # 音声合成
        print("音声合成中...")
        sample_rate, audio_data = self.text_to_speech(response_text)
        
        # 音声再生 (aplay を使用)
        if sample_rate is not None and audio_data is not None:
            print("再生中...")
            self.play_audio_with_aplay()
        else:
            print("音声合成に失敗しました")

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

def main():
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    
    app = Flask(__name__)

    @app.route('/receive_text', methods=['POST'])
    def receive_text():
        data = request.get_json()
        input_text = data.get("text", "")
        if input_text:
            print(f"受信したテキスト: {input_text}")
            pumpkin_talk.process_input_text(input_text)
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"status": "error", "message": "No text provided"}), 400

    print("サーバーを起動します...")
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()