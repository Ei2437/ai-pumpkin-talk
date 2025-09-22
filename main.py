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
        # character.prompt が存在する場合はそれを使用、なければ従来の方法で生成
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
        
        # 録音用のストリームとデータ格納用
        self.recording_stream = None
        self.audio_frames = []
        self.is_recording = False
        self.last_key_state = False  # 前回のキー状態を記録
    
    def match_response_template(self, input_text):
        """入力テキストにマッチする応答テンプレートを探す"""
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
        """応答テキストをフィルタリング"""
        if "response_filtering" in self.advanced_config:
            filtering = self.advanced_config["response_filtering"]
            
            # 除去パターンの適用
            if "remove_patterns" in filtering:
                for pattern in filtering["remove_patterns"]:
                    response_text = re.sub(pattern, "", response_text)
            
            # 置換パターンの適用
            if "replace_patterns" in filtering:
                for old, new in filtering["replace_patterns"].items():
                    response_text = response_text.replace(old, new)
        
        return response_text.strip()
    
    def start_recording(self):
        """録音を開始"""
        if not self.is_recording:
            print("録音開始... (Spaceキーをもう一度押して終了)")
            self.is_recording = True
            self.audio_frames = []
            self.recording_stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.int16)
            self.recording_stream.start()
    
    def stop_recording(self):
        """録音を停止し、AudioDataオブジェクトを返す"""
        if self.is_recording:
            print("録音終了...")
            self.is_recording = False
            self.recording_stream.stop()
            
            # 音声データを結合
            if self.audio_frames:
                audio_data = np.concatenate(self.audio_frames, axis=0)
                # speech_recognition用のAudioDataオブジェクトを作成
                audio = sr.AudioData(audio_data.tobytes(), 16000, 2)
                self.recording_stream.close()
                return audio
            
            self.recording_stream.close()
        return None
    
    def listen_and_transcribe(self):
        """Spaceキーで制御される音声認識（トグル方式）"""
        print("Spaceキーを押して録音開始...")
        
        # ノイズ調整
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            while True:
                current_key_state = keyboard.is_pressed('space')
                
                # キーの状態が変化したときのみ処理
                if current_key_state != self.last_key_state:
                    if current_key_state:  # キーが押された
                        if not self.is_recording:
                            # 録音開始
                            self.start_recording()
                        else:
                            # 録音停止
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
                    # キーが離されたときは何もしない（トグル方式なので）
                    self.last_key_state = current_key_state
                
                # 録音中の場合は音声データを取得
                if self.is_recording:
                    data, overflowed = self.recording_stream.read(1024)
                    if not overflowed:
                        self.audio_frames.append(data)
                
                time.sleep(0.01)  # CPU使用率を下げるための短い待機
                    
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            if self.is_recording:
                self.stop_recording()
            return None
    
    def generate_response(self, input_text):
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        # まずテンプレートマッチを試す
        template_response = self.match_response_template(input_text)
        if template_response:
            return self.filter_response(template_response)
        
        # テンプレートマッチしない場合はAIに生成させる
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
            print(f"生成された回答: {response_text}")
            
            # フィルタリングを適用
            filtered_response = self.filter_response(response_text)
            
            self.conversation_history.append(f"パンプキン: {filtered_response}")
            return filtered_response
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぜ！もう一度話しかけてみろよ！"
    
    def text_to_speech(self, text):
        """VOICEVOXを使用してテキストを音声に変換する"""
        try:
            # 1. テキストから音声合成用のクエリを作成
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            # 2. 音声合成パラメータの調整（system設定から）
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
            
            # 3. 音声合成を実行
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
            
            # WAVデータを読み込む
            wav_data.seek(0)
            sample_rate, audio_data = wavfile.read(wav_data)
            
            # モノラルの場合はステレオに変換
            if len(audio_data.shape) == 1:
                audio_data = np.column_stack((audio_data, audio_data))
            
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None
    
    def play_audio(self, sample_rate, audio_data):
        """音声データを再生する"""
        if sample_rate is None or audio_data is None:
            print("再生できる音声データがありません")
            return
        
        try:
            # 音声の再生
            sd.play(audio_data, sample_rate)
            sd.wait()  # 再生が終わるまで待機
        except Exception as e:
            print(f"音声再生中にエラーが発生しました: {e}")
    
    def run(self):
        """パンプキントークのメインループ"""
        print("=== パンプキントークシステム起動 ===")
        print("俺様、パンプキンの登場だぜ!Spaceキーを押して話しかけてみろよ!")
        print("終了するには Ctrl+C を押してください")
        
        try:
            while True:
                # 1. 音声の文字起こし
                input_text = self.listen_and_transcribe()
                
                if input_text:
                    # 2. 応答の生成
                    response_text = self.generate_response(input_text)
                    
                    # 3. 音声合成
                    print("音声合成中...")
                    sample_rate, audio_data = self.text_to_speech(response_text)
                    
                    # 4. 音声再生
                    print("再生中...")
                    self.play_audio(sample_rate, audio_data)
                
                print("\nSpaceキーを押して次の質問をどうぞ...")
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")

if __name__ == "__main__":
    pumpkin_talk = PumpkinTalk("pumpkin.json")
    pumpkin_talk.run()