import os
import io
import time
import json
import re
import random
import requests
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from flask import Flask, request, jsonify

class LoadConfig:
    def __init__(self, config_path="pumpkin.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
    
    def get_character_prompt(self):
        char = self.config["character"]
        if "prompt" in char:
            # knowledge を結合して prompt を生成
            knowledge_dict = self.config.get("knowledge", {})
            knowledge_str = ""
            for category, items in knowledge_dict.items():
                knowledge_str += f"\n【{category}】\n" + "\n".join(items) + "\n"
            
            return char["prompt"].format(knowledge=knowledge_str)
        else:
            # 旧形式（knowledge が prompt 内にない場合）
            prompt_template = self.config["ai_prompt"]["base_prompt"]
            forbidden_words = "\n".join([f"- {word}" for word in char["speech_style"]["forbidden_words"]])
            endings = "」「".join(char["speech_style"]["sentence_endings"])
            
            return prompt_template.format(
                name=char["name"],
                pronoun=char["pronoun"],
                likes=char["likes"],
                gender=char["gender"],
                personality=char.get("personality", {}).get("description", "横柄で傲慢。いつもはとげとげしているが、甘いものの話題になると急に優しくなる"),
                tone=char["speech_style"]["tone"],
                endings=endings,
                forbidden_words=forbidden_words,
                knowledge=""
            )
    
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

    def filter_response(self, response_text):
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            if "remove_patterns" in filtering:
                for pattern in filtering["remove_patterns"]:
                    response_text = re.sub(pattern, "", response_text)
            
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()

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
            
            filtered_response = self.filter_response(response_text)
            
            self.conversation_history.append(f"パンプキン: {filtered_response}")
            return filtered_response
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぜ！もう一度話しかけてみろよ！"

    def text_to_speech(self, text):
        try:
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            if "voicevox" in self.system_config:
                voicevox_settings = self.system_config["voicevox"]
                if "speed" in voicevox_settings:
                    query_data["speedScale"] = voicevox_settings["speed"]
                if "pitch" in voicevox_settings:
                    query_data["pitchScale"] = voicevox_settings["pitch"]
                if "intonation" in voicevox_settings:
                    query_data["intonationScale"] = voicevox_settings["intonation"]
                if "volume" in voicevox_settings:
                    query_data["volumeScale"] = voicevox_settings["volume"]
                if "post_phoneme_length" in voicevox_settings:
                    query_data["postPhonemeLength"] = voicevox_settings["post_phoneme_length"]
            
            synthesis_url = f"{self.voicevox_url}/synthesis"
            synthesis_params = {"speaker": self.speaker_id}
            synthesis_response = requests.post(
                synthesis_url, 
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"}
            )
            synthesis_response.raise_for_status()
            
            wav_data = io.BytesIO(synthesis_response.content)
            wav_data.seek(0)
            sample_rate, audio_data = wavfile.read(wav_data)
            
            if len(audio_data.shape) == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None

    def play_audio(self, sample_rate, audio_data):
        if sample_rate is None or audio_data is None:
            print("再生できる音声データがありません")
            return
        
        try:
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"音声再生中にエラーが発生しました: {e}")

    def process_input_text(self, input_text):
        # 応答の生成
        response_text = self.generate_response(input_text)
        print("回答:", response_text)
        
        # 音声合成
        print("音声合成中...")
        sample_rate, audio_data = self.text_to_speech(response_text)
        
        # 音声再生
        print("再生中...")
        self.play_audio(sample_rate, audio_data)

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