
# --- 修正版 ver 1.4: プロキシ無効化対応 ---

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

# ===== プロキシ完全無効化 =====
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
        if "prompt" in char:
            return char["prompt"]
        else:
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
                forbidden_words=forbidden_words
            )
    
    def get_ollama_config(self):
        return self.config["api"]["ollama"]
    
    def get_voicevox_config(self):
        return self.config["api"]["voicevox"]
    
    def get_response_templates(self):
        return self.config.get("response_templates", {})
    
    def get_system_config(self):
        return self.config.get("system", {})
    
    def get_advanced_config(self):
        return self.config.get("advanced", {})


class PumpkinTalk:
    def __init__(self, config_path="pumpkin.json"):
        self.config_loader = LoadConfig(config_path)
        self.ollama_config = self.config_loader.get_ollama_config()
        self.voicevox_config = self.config_loader.get_voicevox_config()
        self.response_templates = self.config_loader.get_response_templates()
        self.system_config = self.config_loader.get_system_config()
        self.advanced_config = self.config_loader.get_advanced_config()
        
        self.recognizer = sr.Recognizer()
        self.ollama_url = self.ollama_config["url"]
        self.voicevox_url = self.voicevox_config["url"]
        self.speaker_id = self.voicevox_config["speaker_id"]
        self.model = self.ollama_config["model"]
        
        self.character_prompt = self.config_loader.get_character_prompt()
        self.conversation_history = []
    
    def match_response_template(self, input_text):
        best_match = None
        best_priority = -1
        for template_name, template in self.response_templates.items():
            for pattern in template["patterns"]:
                if re.search(pattern, input_text, re.IGNORECASE):
                    priority = template.get("priority", 0)
                    if priority > best_priority:
                        best_priority = priority
                        best_match = template
                    break
        if best_match:
            responses = best_match["responses"]
            return random.choice(responses)
        return None
    
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
    
    def listen_and_transcribe(self):
        print("🎤 Enterキーを押して話してください...")
        input(">> ")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("録音中... 話し終わったら自動で停止します")
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                print("文字起こし中...")
                text = self.recognizer.recognize_google(audio, language="ja-JP")
                print(f"認識されたテキスト: {text}")
                return text
            except sr.UnknownValueError:
                print("音声を認識できませんでした")
                return None
            except sr.RequestError as e:
                print(f"音声認識サービスでエラーが発生しました: {e}")
                return None
    
    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        template_response = self.match_response_template(input_text)
        if template_response:
            return self.filter_response(template_response)
        
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
            query_response = requests.post(query_url, params=query_params, proxies=REQUESTS_NO_PROXY)
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
                headers={"Content-Type": "application/json"},
                proxies=REQUESTS_NO_PROXY
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
    
    def run(self):
        print("=== activate ===")
        try:
            while True:
                input_text = self.listen_and_transcribe()
                if input_text:
                    response_text = self.generate_response(input_text)
                    print("回答:", response_text)
                    print("音声合成中...")
                    sample_rate, audio_data = self.text_to_speech(response_text)
                    print("再生中...")
                    self.play_audio(sample_rate, audio_data)
                print("\nNext...")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")


if __name__ == "__main__":
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    pumpkin_talk.run()
