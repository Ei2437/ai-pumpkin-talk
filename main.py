import speech_recognition as sr
import requests
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import time
import io

class PumpkinTalk:
    def __init__(self, ollama_url="http://localhost:11434", voicevox_url="http://localhost:50021"):
        self.recognizer = sr.Recognizer()
        self.ollama_url = ollama_url
        self.voicevox_url = voicevox_url
        
        # VOICEVOXのスピーカーID設定（パンプキンキャラに合う声を選択）
        # 例: 九州そら ノーマル(16)、ささっき(19)、玄野武宏(11)など
        self.speaker_id = 1  # 玄野武宏（少し低めの男性的な声）
        
        # キャラクター設定
        self.character_name = "パンプキン"
        self.character_prompt = """
        あなたは「パンプキン」というキャラクターです。以下の特徴に従って応答してください：

        【重要：必ず日本語のみで応答すること。中国語を混ぜないこと。】

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
    
    def listen_and_transcribe(self):
        """マイクからの音声を取得し、文字起こしを行う"""
        print("聞き取り中... (話し始めてください)")
        
        with sr.Microphone() as source:
            # ノイズ調整
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                # 音声の取得
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
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
        """Ollamaを使用して応答を生成する"""
        if not input_text:
            return "何か言ったか？もう一度言ってみろよ！"
        
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
                    "temperature": 0.7,  # 少し創造性を高める
                    "top_p": 0.95
                }
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # レスポンスから回答テキストを抽出
            result = response.json()
            response_text = result.get("response", "応答を生成できませんでした。")
            print(f"生成された回答: {response_text}")
            
            # 会話履歴に応答を追加
            self.conversation_history.append(f"パンプキン: {response_text}")
            
            return response_text
            
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
        print("俺様、パンプキンの登場だぜ！何か質問があるなら言ってみろよ！")
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
                
                print("\n次の質問をどうぞ...")
                time.sleep(1)  # 少し間を置く
                
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")  # 終了メッセージ修正

if __name__ == "__main__":
    # Ollamaサーバーのアドレスとポート
    ollama_url = "http://localhost:11434"
    
    # VOICEVOXサーバーのアドレスとポート
    voicevox_url = "http://localhost:50021"
    
    # システム起動
    pumpkin_talk = PumpkinTalk(ollama_url, voicevox_url)
    pumpkin_talk.run()



    # .\venv\Scripts\activate
>>>>>>> 9fc5140 (Auto Update 2025-09-20 19:59:51)
    # deactivate