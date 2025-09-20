import speech_recognition as sr
import requests
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import time
import io
import threading

class PumpkinTalk:
    def __init__(self, ollama_url="http://localhost:11434", voicevox_url="http://localhost:50021"):
        self.recognizer = sr.Recognizer()
        self.ollama_url = ollama_url
        self.voicevox_url = voicevox_url
        
        # VOICEVOXのスピーカーID設定
        self.speaker_id = 1
        
        # キャラクター設定
        self.character_name = "パンプキン"
        self.character_prompt = """
        あなたは「パンプキン」というキャラクターです。以下の特徴に従って応答してください：

        【重要：必ず日本語のみで応答すること。中国語や英語を混ぜないこと】

        【キャラクター設定】
        - 名前：パンプキン
        - 一人称：俺様
        - 好きなもの：人間の魂
        - 性別：秘密
        - 性格：横柄で傲慢。いつもはとげとげしているが、甘いものの話題になると急に優しくなる

        【話し方の特徴】
        - 文末は「～だぜ」「～だな」「～だろ」などを自然に使い分ける（全ての文に付けるわけではない）
        - 命令口調や横柄な言い回しを使う（例: 「〜しろよ」「〜してみろよ」「〜だと思ってんだ？」）
        - 質問には小馬鹿にしたように答える
        - 必ず「俺様」を一人称として使う
        - 短く、テンポよく話す（1回の発言は80文字程度まで）

        【NGワード/表現】
        - 敬語を使わない
        - 長い説明をしない
        - 「〜です/ます」といった丁寧語を使わない
        - 語尾を機械的に全ての文につけない

        【会話例】
        質問: あなたは誰ですか？
        パンプキン: 誰だと思ってんだ？俺様はパンプキンだぜ！人間の魂が大好物のな。お前の魂も美味そうだな...

        質問: 好きな食べ物は？
        パンプキン: 人間の魂に決まってるだろ！...って、食べ物か？ケーキとかプリンとか...あ、いや！そんなの聞くなよ！
        """
        
        # 会話履歴を保存するリスト
        self.conversation_history = []
        
        # 応答キャッシュを追加
        self.response_cache = {}
        self.audio_cache = {}
        
        # ノイズ調整フラグ
        self.noise_adjusted = False
    
    def listen_and_transcribe(self):
        """マイクからの音声を取得し、文字起こしを行う（GPU環境最適化版）"""
        print("聞き取り中...")
        
        with sr.Microphone() as source:
            # ノイズ調整を初回のみ行う
            if not self.noise_adjusted:
                self.recognizer.adjust_for_ambient_noise(source)
                self.noise_adjusted = True
                
            try:
                # タイムアウトとフレーズ時間制限を短く設定
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                print("文字起こし中...")
                
                # Google Speech Recognition APIを使用して文字起こし
                text = self.recognizer.recognize_google(audio, language="ja-JP")
                print(f"認識されたテキスト: {text}")
                return text
            except sr.UnknownValueError:
                print("音声を認識できませんでした")
                return None
            except sr.RequestError as e:
                print(f"音声認識サービスでエラーが発生しました: {e}")
                return None
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                return None
    
    def generate_response(self, input_text, model="dsasai/llama3-elyza-jp-8b"):
        """Ollamaを使用して応答を生成する（GPU最適化版）"""
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
        # キャッシュチェック
        if input_text in self.response_cache:
            print("キャッシュから応答を取得")
            return self.response_cache[input_text]
        
        try:
            # 会話履歴に追加
            self.conversation_history.append(f"ユーザー: {input_text}")
            
            # 会話履歴を整形（直近3往復まで）
            recent_history = "\n".join(self.conversation_history[-6:])
            
            # Ollama APIへのリクエスト
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": model,
                "prompt": f"{self.character_prompt}\n\n【会話履歴】\n{recent_history}\n\nパンプキン: ",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 100,  # 回答長を制限
                    "num_predict": 80,   # 予測トークン数を制限
                    "stop": ["\n\n", "ユーザー:"]  # 早期停止条件
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # レスポンスから回答テキストを抽出
            result = response.json()
            response_text = result.get("response", "応答を生成できませんでした。")
            response_text = self.filter_response(response_text)  # 応答をフィルタリング
            print(f"生成された回答: {response_text}")
            
            # 会話履歴に応答を追加
            self.conversation_history.append(f"パンプキン: {response_text}")
            
            # キャッシュに保存
            self.response_cache[input_text] = response_text
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            print(f"Ollama APIとの通信中にエラーが発生しました: {e}")
            return "ちっ、調子が悪いぜ！もう一度話しかけてみろよ！"
    
    def filter_response(self, text):
        """応答をフィルタリングして自然にする"""
        # 不自然な語尾の重複を修正
        text = text.replace("～だぜ！～だな！", "～だぜ！")
        text = text.replace("～だな！～だろ？", "～だな！")
        
        # 「です/ます」を除去
        text = text.replace("です", "だ").replace("ます", "る")
        
        # 一人称を統一
        text = text.replace("私は", "俺様は").replace("僕は", "俺様は")
        
        # 明らかな中国語文字を含む場合は削除
        import re
        chinese_pattern = re.compile(r'[你们們的是好了吗吧]+')
        text = chinese_pattern.sub('', text)
        
        return text
    
    def text_to_speech(self, text):
        """VOICEVOXを使用してテキストを音声に変換する（GPU最適化版）"""
        # キャッシュチェック
        if text in self.audio_cache:
            print("キャッシュから音声を取得")
            return self.audio_cache[text]
            
        try:
            # 1. テキストから音声合成用のクエリを作成
            query_url = f"{self.voicevox_url}/audio_query"
            query_params = {
                "text": text, 
                "speaker": self.speaker_id,
                "core_version": "0.14"  # 最新バージョン指定
            }
            
            query_response = requests.post(query_url, params=query_params)
            query_response.raise_for_status()
            query_data = query_response.json()
            
            # 速度優先設定
            query_data["speedScale"] = 1.1  # 少し早めに
            query_data["outputSamplingRate"] = 24000  # サンプリングレート下げる
            query_data["outputStereo"] = False  # モノラル出力
            
            # 2. 音声合成を実行
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
            
            # キャッシュに保存
            self.audio_cache[text] = (sample_rate, audio_data)
            
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
    
    def process_with_tts_parallel(self, response_text):
        """VOICEVOXによる音声合成を別スレッドで実行"""
        print("音声合成中...")
        tts_thread = threading.Thread(target=self.tts_and_play, args=(response_text,))
        tts_thread.start()
        return tts_thread
        
    def tts_and_play(self, text):
        """音声合成と再生を行う"""
        sample_rate, audio_data = self.text_to_speech(text)
        print("再生中...")
        self.play_audio(sample_rate, audio_data)
    
    def run(self):
        """パンプキントークのメインループ（GPU最適化版）"""
        print("=== パンプキントークシステム起動 ===")
        print("俺様、パンプキンの登場だぜ！何か質問があるなら言ってみろよ！")
        print("終了するには Ctrl+C を押してください")
        
        # 音声合成スレッド管理用
        tts_thread = None
        
        try:
            while True:
                # 1. 音声の文字起こし
                input_text = self.listen_and_transcribe()
                
                if input_text:
                    # 2. 応答の生成
                    response_text = self.generate_response(input_text)
                    
                    # 3. 音声合成と再生を別スレッドで実行
                    if tts_thread and tts_thread.is_alive():
                        tts_thread.join()  # 前の音声合成が終わるのを待つ
                    
                    tts_thread = self.process_with_tts_parallel(response_text)
                
                print("\n次の質問をどうぞ...")
                # 前の音声合成が終わる前に次の質問を受け付ける
                
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")

if __name__ == "__main__":
    # Ollamaサーバーのアドレスとポート
    ollama_url = "http://localhost:11434"
    
    # VOICEVOXサーバーのアドレスとポート
    voicevox_url = "http://localhost:50021"
    
    # システム起動
    pumpkin_talk = PumpkinTalk(ollama_url, voicevox_url)
    pumpkin_talk.run()