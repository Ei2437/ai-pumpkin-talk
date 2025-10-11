# ver1.3  10/10 17:34

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
import keyboard


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
        
        self.recognizer = sr.Recognizer()
        self.ollama_url = self.ollama_config["url"]
        self.voicevox_url = self.voicevox_config["url"]
        self.speaker_id = self.voicevox_config["speaker_id"]
        self.model = self.ollama_config["model"]
        
        self.character_prompt = self.config_loader.get_character_prompt()
        self.conversation_history = []
        
        # 録音用
        self.recording_stream = None
        self.audio_frames = []
        self.is_recording = False
        self.last_key_state = False
    
    def filter_response(self, response_text):
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            # 除去
            if "remove_patterns" in filtering:
                for pattern in filtering["remove_patterns"]:
                    response_text = re.sub(pattern, "", response_text)
            
            # 置換
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()
    
    def start_recording(self):
        if not self.is_recording:
            print("録音開始...")
            self.is_recording = True
            self.audio_frames = []
            self.recording_stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.int16)
            self.recording_stream.start()
    
    def stop_recording(self):
        if self.is_recording:
            print("録音終了...")
            self.is_recording = False
            self.recording_stream.stop()
            
            # 音声データを結合
            if self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                # 文字起こし用のAudioDataオブジェクトを作成
                audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
                self.recording_stream.close()
                # AudioDataオブジェクトを返す
                return audio
            
            self.recording_stream.close()
        return None
    
    def listen_and_transcribe(self):
        # ノイズ調整
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            while True:
                current_key_state = keyboard.is_pressed('space')
                
                # キーの状態が変化したときのみ処理
                if current_key_state != self.last_key_state:
                    if current_key_state: #押された
                        if not self.is_recording:
                            # 開始
                            self.start_recording()
                        else:
                            # 停止
                            audio = self.stop_recording()
                            if audio:
                                try:
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
                    self.last_key_state = current_key_state
                
                # 録音中の場合は音声データを取得
                if self.is_recording:
                    data, overflowed = self.recording_stream.read(1024)
                    if not overflowed:
                        self.audio_frames.append(data)
                
                time.sleep(0.01)
                    
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            if self.is_recording:
                self.stop_recording()
            return None
    
    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        # テンプレートマッチングを削除（最新版では使用しない）
        # 代わりに、AI に knowledge をもとに応答させる
        
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
        print("Spaceキーを押して録音開始...")
        try:
            while True:
                # 文字起こし
                input_text = self.listen_and_transcribe()
                
                if input_text:
                    # 応答の生成
                    response_text = self.generate_response(input_text)
                    print("回答:", response_text)
                    # 音声合成
                    print("音声合成中...")
                    sample_rate, audio_data = self.text_to_speech(response_text)
                    
                    # 音声再生
                    print("再生中...")
                    self.play_audio(sample_rate, audio_data)
                
                print("\nNext...")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")

if __name__ == "__main__":
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    pumpkin_talk.run()

# コードの解説はREADME.mdを見てください。

# --- 更新内容 ---
# ver 1.0  -  prototype.pyのプロンプト形式を一新し、README.mdに記載した形式でpumpkin.jsonに統合。
# ver 1.1  -  Ollamaを sudume の gemma3:latest に変更。
# ver 1.2  -  音声合成をsudume側のVOICEVOXに変更。
# ver 1.3  -  response_templates を削除し、knowledge を使用する形式に変更。