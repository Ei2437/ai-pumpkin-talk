# ver 1.3  9/23 

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
from pynput import keyboard


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
        
        # 録音用
        self.recording_stream = None
        self.audio_frames = []
        self.is_recording = False
        
        # キーボード入力の設定
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()
        self.Q_pressed = False
    
    def on_key_press(self, key):
        try:
            if key.char == 'q':
                self.Q_pressed = True
        except AttributeError:
            # 特殊キー（Ctrl, Altなど）は無視
            pass
    
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
        
        print("録音を開始するにはQキーを押してください...")
        
        try:
            while True:
                if self.Q_pressed:
                    self.Q_pressed = False  # フラグリセット
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
                
                # 録音中の場合は音声データを取得
                if self.is_recording:
                    data, overflowed = self.recording_stream.read(1024)
                    if not overflowed:
                        self.audio_frames.append(data)
                
                time.sleep(0.01)  # CPU負荷軽減
                    
        except KeyboardInterrupt:
            print("\n録音を中断しました。")
            if self.is_recording:
                self.stop_recording()
            return None
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            if self.is_recording:
                self.stop_recording()
            return None
    
    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        # まずはテンプレから探す
        template_response = self.match_response_template(input_text)
        if template_response:
            return self.filter_response(template_response)
        
        # ない場合はOllamaで生成
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
        print("Qキーを押して録音開始...")
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
        
        # キーボードリスナーを停止
        self.listener.stop()


if __name__ == "__main__":
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    pumpkin_talk.run()

# コードの解説はREADME.mdを見てください。
# sudume の Ollama を使用する際は gemma3:latest 一択。

# 今文字起こしに使用している Google Web Speech API を使用する際、「SpeechRecognition」というモジュールを必要とする。
# そして、そのモジュールの動作に、内部的に「pyaudio」を使用しているため、pyaudio が必須。
# pyaudioを入れるためにはビルドするための関連モジュールやヘッダーファイルが欲しいため、portaudio19-devが必須。
# また、音声の入出力に使用する「sounddevice」には portaudio19-dev が必須。
# sudo apt install portaudio19-dev
# pip install pyaudio


# --- 更新内容 ---
# ver 1.0  -  prototype.pyのプロンプト形式を一新し、README.mdに記載した形式でpumpkin.jsonに統合。
# ver 1.1  -  Ollamaを sudume の gemma3:latest に変更。
# ver 1.2  -  Ubuntu対応版(仮)
# ver 1.3  -  pynput分岐を削除