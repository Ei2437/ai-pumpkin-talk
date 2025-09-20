import speech_recognition as sr
import requests
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import time
import io
import threading
import re
import keyboard
import os
import pyaudio
import wave
import tempfile

class AdvancedConfigManager:
    """正規表現パターンによる高度なテンプレートマッチングを提供するクラス"""
    
    def __init__(self, config_file="pumpkin_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.used_templates = set()  # 使用済みテンプレートを追跡
        self.compiled_patterns = {}  # コンパイル済み正規表現
        self.compiled_negatives = {}  # コンパイル済み否定パターン
        self._compile_patterns()
    
    def load_config(self):
        """設定ファイルを読み込む"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"設定ファイル '{self.config_file}' を読み込みました")
                return config
        except FileNotFoundError:
            print(f"設定ファイル '{self.config_file}' が見つかりません")
            raise
        except json.JSONDecodeError:
            print(f"設定ファイル '{self.config_file}' の形式が正しくありません")
            raise
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {e}")
            raise
    
    def _compile_patterns(self):
        """正規表現パターンをコンパイルする"""
        if "pattern_matching" not in self.config:
            # 後方互換性のため、キーワードベースのマッチングをサポート
            return
            
        # 各カテゴリのパターンをコンパイル
        for category, pattern_data in self.config["pattern_matching"].items():
            if "patterns" in pattern_data:
                self.compiled_patterns[category] = [
                    re.compile(p, re.IGNORECASE) for p in pattern_data["patterns"]
                ]
            
            # 否定パターンがあれば、それもコンパイル
            if "negative_patterns" in pattern_data:
                self.compiled_negatives[category] = [
                    re.compile(p, re.IGNORECASE) for p in pattern_data["negative_patterns"]
                ]
    
    def get_prompt(self):
        """プロンプトを取得"""
        return self.config.get("character_prompt", "")
    
    def get_template(self, category):
        """カテゴリから未使用のテンプレートを取得"""
        if "templates" not in self.config or category not in self.config["templates"] or not self.config["templates"][category]:
            return None
        
        # 使用可能なテンプレート（まだ使われていないもの）をフィルタリング
        available_templates = [t for t in self.config["templates"][category] 
                            if f"{category}:{t}" not in self.used_templates]
        
        # 使用可能なテンプレートがない場合は全てリセットして最初から
        if not available_templates:
            for used_key in list(self.used_templates):
                if used_key.startswith(f"{category}:"):
                    self.used_templates.remove(used_key)
            available_templates = self.config["templates"][category]
        
        # ランダムに選択
        import random
        template = random.choice(available_templates)
        self.used_templates.add(f"{category}:{template}")
        return template
    
    def find_matching_category(self, query):
        """パターンベースのマッチングでカテゴリを特定"""
        # パターンマッチングが設定されている場合はそれを使用
        if hasattr(self, 'compiled_patterns') and self.compiled_patterns:
            scores = self._calculate_pattern_scores(query)
            if scores:
                # 最高スコアのカテゴリを返す
                best_category = max(scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:  # スコアが正の場合のみ
                    return best_category[0]
            
        # パターンマッチングでヒットしなかった場合は従来のキーワードマッチングを試す
        return self._legacy_keyword_matching(query)
    
    def _calculate_pattern_scores(self, query):
        """パターンマッチングのスコアを計算"""
        scores = {}
        
        # 各カテゴリのパターンでマッチングを試みる
        for category, patterns in self.compiled_patterns.items():
            scores[category] = 0
            for pattern in patterns:
                if pattern.search(query):
                    scores[category] += 1
        
        # 否定パターンを適用（スコアを減算）
        for category, patterns in self.compiled_negatives.items():
            if category in scores:
                for pattern in patterns:
                    if pattern.search(query):
                        scores[category] -= 2  # 否定パターンはより強い効果
        
        return scores
    
    def _legacy_keyword_matching(self, query):
        """従来のキーワードベースのマッチング（後方互換性用）"""
        if "keywords" not in self.config:
            return None
        
        # 質問を小文字化して検索を容易に
        query_lower = query.lower()
        
        # 各カテゴリのキーワードとマッチするか確認
        for category, words in self.config["keywords"].items():
            for word in words:
                if word in query_lower:
                    return category
        
        return None
    
    def get_setting(self, key, default_value=None):
        """設定値を取得"""
        if "settings" not in self.config:
            return default_value
        return self.config["settings"].get(key, default_value)


class PumpkinTalk:
    def __init__(self, ollama_url="http://localhost:11434", voicevox_url="http://localhost:50021", config_file="pumpkin_config.json"):
        self.recognizer = sr.Recognizer()
        self.ollama_url = ollama_url
        self.voicevox_url = voicevox_url
        
        # 高度な設定マネージャーの初期化
        self.config_manager = AdvancedConfigManager(config_file)
        
        # 設定から値を取得
        self.speaker_id = self.config_manager.get_setting("speaker_id", 1)
        self.model_name = self.config_manager.get_setting("model_name", "dsasai/llama3-elyza-jp-8b")
        self.character_name = "パンプキン"
        
        # キャラクター設定（外部ファイルから取得）
        self.character_prompt = self.config_manager.get_prompt()
        
        self.conversation_history = []
        self.response_cache = {}
        self.audio_cache = {}
        self.noise_adjusted = False
        self.audio_thread = None
        self.is_speaking = False
        self.audio_completed = threading.Event()  # 音声再生完了を追跡するイベント
        
        # 録音関連の設定
        self.recording_stopped = False
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.p = pyaudio.PyAudio()
    
    def wait_for_key_press(self, key="space"):
        """指定されたキーが押されるまで待機"""
        print(f"\n[{key.upper()}]キーを押して質問を開始...")
        keyboard.wait(key)
        print("\nキーが押されました。質問をどうぞ...")
    
    def record_with_key_control(self):
        """スペースキーで録音開始と終了を制御する"""
        # スペースキーを押して録音を開始
        self.wait_for_key_press("space")
        
        # 一時ファイルを作成
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wave_file = wave.open(temp_file.name, 'wb')
        wave_file.setnchannels(self.CHANNELS)
        wave_file.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wave_file.setframerate(self.RATE)
        
        # マイクストリームを開く
        stream = self.p.open(format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK)
        
        print("録音中... [SPACE]キーを押して録音を終了")
        
        frames = []
        self.recording_stopped = False
        
        # スペースキーが押されたら録音を停止するフック
        def on_key_press(e):
            if e.name == 'space':
                self.recording_stopped = True
                print("\n録音を終了します...")
        
        # キーボードフックを登録
        keyboard.on_press(on_key_press)
        
        try:
            # 録音ループ
            while not self.recording_stopped:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
        finally:
            # キーボードフックを解除
            keyboard.unhook_all()
            
            # ストリームを閉じる
            stream.stop_stream()
            stream.close()
            
            # 音声データを保存
            if frames:
                for frame in frames:
                    wave_file.writeframes(frame)
            wave_file.close()
            
            file_path = temp_file.name
            return file_path
    
    def transcribe_audio_file(self, audio_file_path):
        """音声ファイルを文字起こし"""
        print("文字起こし中...")
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language="ja-JP")
                print(f"認識されたテキスト: {text}")
                return text
        except sr.UnknownValueError:
            print("音声を認識できませんでした")
            return None
        except sr.RequestError as e:
            print(f"音声認識サービスでエラーが発生しました: {e}")
            return None
        except Exception as e:
            print(f"文字起こし中にエラーが発生しました: {e}")
            return None
        finally:
            # 一時ファイルを削除
            try:
                os.unlink(audio_file_path)
            except:
                pass
    
    def listen_and_transcribe(self):
        """キー制御で録音し、文字起こしを行う"""
        # 音声再生中は録音しない
        if self.is_speaking:
            print("音声再生中です。完了までお待ちください...")
            return None
            
        # スペースキーで録音開始・終了
        audio_file_path = self.record_with_key_control()
        
        # 録音された音声を文字起こし
        text = self.transcribe_audio_file(audio_file_path)
        return text
    
    def filter_response(self, text):
        """応答を後処理して自然にする"""
        text = text.replace("～だぜ！～だな！", "～だぜ！")
        text = text.replace("～だな！～だろ？", "～だな！")
        text = text.replace("です", "だ").replace("ます", "る")
        text = text.replace("私は", "俺様は").replace("僕は", "俺様は")
        chinese_pattern = re.compile(r'[你们們的是好了吗吧]+')
        text = chinese_pattern.sub('', text)
        return text
    
    def generate_response(self, input_text):
        """応答を生成する（テンプレート優先）"""
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"

        # キャッシュチェック
        if input_text in self.response_cache:
            print("キャッシュから応答を取得")
            return self.response_cache[input_text]
        
        # パターンマッチングでカテゴリを特定
        category = self.config_manager.find_matching_category(input_text)
        if category:
            template_response = self.config_manager.get_template(category)
            if template_response:
                print(f"テンプレート({category})から応答を取得")
                # 会話履歴に追加
                self.conversation_history.append(f"ユーザー: {input_text}")
                self.conversation_history.append(f"パンプキン: {template_response}")
                # キャッシュに保存
                self.response_cache[input_text] = template_response
                return template_response
        
        try:
            # テンプレートがない場合はAIで生成
            self.conversation_history.append(f"ユーザー: {input_text}")
            
            recent_history = "\n".join(self.conversation_history[-6:])

            # 設定から値を取得
            temperature = self.config_manager.get_setting("temperature", 0.7)
            top_p = self.config_manager.get_setting("top_p", 0.95)
            max_tokens = self.config_manager.get_setting("max_tokens", 150)
            num_predict = self.config_manager.get_setting("num_predict", 80)

            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "num_predict": num_predict
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "応答を生成できませんでした。")
            response_text = self.filter_response(response_text)
            print(f"生成された回答: {response_text}")
            
            self.conversation_history.append(f"パンプキン: {response_text}")
            self.response_cache[input_text] = response_text
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぜ！もう一度話しかけてみろよ！"
    
    def text_to_speech(self, text):
        """VOICEVOXを使用してテキストを音声に変換する"""
        if text in self.audio_cache:
            print("キャッシュから音声を取得")
            return self.audio_cache[text]
            
        try:
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {"text": text, "speaker": self.speaker_id}
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            query_data["speedScale"] = 1.1  
            query_data["outputSamplingRate"] = 24000
            query_data["outputStereo"] = False
            
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
            
            self.audio_cache[text] = (sample_rate, audio_data)
            return sample_rate, audio_data
            
        except requests.exceptions.RequestException as e:
            print(f"VOICEVOX APIとの通信中にエラーが発生しました: {e}")
            return None, None
    
    def play_audio(self, sample_rate, audio_data):
        """音声データを再生する"""
        if sample_rate is None or audio_data is None:
            print("再生できる音声データがありません")
            self.audio_completed.set()  # イベントを設定して完了を通知
            return
        
        try:
            self.is_speaking = True
            self.audio_completed.clear()  # イベントをクリア
            sd.play(audio_data, sample_rate)
            sd.wait()  # 再生が終わるまで待機
            self.is_speaking = False
            print("----- 音声再生完了 -----")
            self.audio_completed.set()  # イベントを設定して完了を通知
        except Exception as e:
            print(f"音声再生中にエラーが発生しました: {e}")
            self.is_speaking = False
            self.audio_completed.set()  # エラー時にもイベントを設定
    
    def process_audio_thread(self, text):
        """音声合成と再生を行う（スレッド用）"""
        try:
            sample_rate, audio_data = self.text_to_speech(text)
            print("再生中...")
            self.play_audio(sample_rate, audio_data)
        except Exception as e:
            print(f"音声処理中にエラーが発生しました: {e}")
            self.is_speaking = False
            self.audio_completed.set()  # エラー時にもイベントを設定
    
    def cleanup(self):
        """リソースを解放"""
        if hasattr(self, 'p') and self.p:
            self.p.terminate()
    
    def run(self):
        """パンプキントークのメインループ"""
        print("=== パンプキントーク パターンマッチング版 ===")
        print("俺様、パンプキンの登場だぜ！何か質問があるなら言ってみろよ！")
        print("質問するには[SPACE]キーを押してから話し、終了時も[SPACE]キーを押してください")
        print("終了するには Ctrl+C を押してください")
        
        # 最初のイベントを設定
        self.audio_completed.set()
        
        try:
            while True:
                # 音声再生が完了するまで待機
                if not self.audio_completed.is_set():
                    print("音声再生の完了を待機中...")
                    self.audio_completed.wait()
                
                # 音声再生が完了した後に次の質問を受け付ける
                input_text = self.listen_and_transcribe()
                
                if input_text:
                    response_text = self.generate_response(input_text)
                    
                    print("音声合成中...")
                    
                    # 前の音声処理スレッドが終了するのを待つ
                    if self.audio_thread and self.audio_thread.is_alive():
                        self.audio_thread.join()
                    
                    # 新しいスレッドで音声処理を実行
                    self.audio_thread = threading.Thread(target=self.process_audio_thread, args=(response_text,))
                    self.audio_thread.daemon = True
                    self.audio_thread.start()
                
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1)
        finally:
            self.cleanup()

if __name__ == "__main__":
    # 設定ファイルのパスを指定（デフォルトは "pumpkin_config.json"）
    config_file = "pumpkin_config.json"
    
    try:
        pumpkin_talk = PumpkinTalk(config_file=config_file)
        pumpkin_talk.run()
    except FileNotFoundError:
        print(f"エラー: 設定ファイル '{config_file}' が見つかりません。")
        print("正しい設定ファイルを用意してから再度実行してください。")
    except Exception as e:
        print(f"エラー: {e}")