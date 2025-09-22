# コード解説 & json仕様解説

main.py 及び pumpkin.json の説明です。Windowsでの実行環境構築に関しては how_to.txt を見てください。

## main.py

### LoadConfig クラス
設定ファイル(pumpkin.json)を読み込むクラス

#### `__init__(self, config_path="pumpkin.json")`
- 設定ファイルのパスを受け取る（デフォルトは"pumpkin.json"）
- ファイルが存在しない場合はエラーを発生
- JSONファイルを読み込んで`self.config`に格納

#### `get_character_prompt(self)`
- キャラクターのプロンプトを取得
- `character.prompt`が直接定義されていればそれを返す
- 定義されていなければ、テンプレートから動的に生成

#### `get_ollama_config(self)`
- Ollama APIの設定を取得して返す

#### `get_voicevox_config(self)`
- VOICEVOX APIの設定を取得して返す

#### `get_response_templates(self)`
- 応答テンプレートの設定を取得して返す（ない場合は空の辞書）

#### `get_system_config(self)`
- システム詳細設定を取得して返す（ない場合は空の辞書）

#### `get_advanced_config(self)`
- 高度な設定を取得して返す（ない場合は空の辞書）

### PumpkinTalk クラス
パンプキントークシステムのメイン処理を行うクラス

#### `__init__(self, config_path="pumpkin.json")`
- LoadConfigを使って設定を読み込む
- 各種API設定、キャラクタープロンプトを初期化
- 録音関連の変数を初期化
- 会話履歴のリストを初期化

#### `match_response_template(self, input_text)`
- 入力テキストにマッチする応答テンプレートを探す
- 各テンプレートのパターンを正規表現でチェック
- 優先度が高いものを選択
- 見つかったらランダムに応答を選んで返す
- 見つからなければNoneを返す

#### `filter_response(self, response_text)`
- 応答テキストにフィルタリングを適用
- 削除パターンにマッチする部分を削除
- 置換パターンに従ってテキストを置換
- 不要な空白を削除して返す

#### `start_recording(self)`
- 録音を開始する
- すでに録音中でなければ、新しい録音ストリームを開始
- 音声データ格納用のリストを初期化

#### `stop_recording(self)`
- 録音を停止する
- 録音中の場合、ストリームを停止
- 収集した音声データを結合してAudioDataオブジェクトを作成
- AudioDataオブジェクトを返す

#### `listen_and_transcribe(self)`
- Spaceキーで制御される音声認識処理
- Spaceキーの押下状態を監視（トグル方式）
- 最初の押下で録音開始、次の押下で録音停止
- 録音停止後、Google音声認識APIで文字起こし
- 認識されたテキストを返す

#### `generate_response(self, input_text)`
- 入力テキストに対する応答を生成
- まずテンプレートマッチを試す
- マッチすればフィルタリングして返す
- マッチしなければOllama APIでAI生成
- 会話履歴を更新して応答を返す

#### `text_to_speech(self, text)`
- VOICEVOXを使ってテキストを音声に変換
- 音声合成クエリを作成
- システム設定のパラメータを適用
- 音声合成を実行
- 音声データを取得して返す

#### `play_audio(self, sample_rate, audio_data)`
- 音声データを再生する
- サンプルレートと音声データを受け取る
- sounddeviceを使って音声を再生
- 再生完了まで待機

#### `run(self)`
- パンプキントークシステムのメインループ
- 無限ループで以下の処理を繰り返す：
  1. 音声の文字起こし
  2. 応答の生成
  3. 音声合成
  4. 音声再生
- Ctrl+Cで終了

#### `if __name__ == "__main__":`
- スクリプトが直接実行された場合のエントリーポイント
- PumpkinTalkインスタンスを作成して実行

## pumpkin.json

### 1. Character
- name      キャラクター名（パンプキン） 
- pronoun   一人称（俺様）
- likes     好きなもの（人間の魂）
- gender    性別（秘密）
- prompt    キャラクターの詳細プロンプト

### 2. Personality
- base_mood: 基本ムード（arrogant）  
- trigger_moods: トリガームード  
- sweets: 甘いもの（friendly）  
- technology: 技術（curious）  
- souls: 魂（excited）  

### 3. Speech Style
- sentence_endings: 文末表現のリスト  
- forbidden_words: 禁止語のリスト  
- tone: 語調（短く、テンポよく）  

### 4. AI Prompt
- base_prompt: ベースとなるプロンプトテンプレート  

### 5. API
#### Ollama
- url: APIエンドポイント  
- model: 使用モデル  
- params: パラメータ設定（temperature, top_p）
#### Voicevox
- url: APIエンドポイント  
- speaker_id: スピーカーID  

### 6. Response Templates
#### カテゴリ
- introduction: 自己紹介関連  
- preferences: 好み関連  
- school: 学校関連  
- programming: プログラミング関連  
- anime: アニメ関連  
- greeting: 挨拶  
- farewell: 別れ  
#### 各テンプレートの機能
- patterns: マッチングパターン（正規表現）  
- responses: 応答例のリスト  
- priority: 優先度  

### 7. System
#### Voicevox
- url: APIエンドポイント  
- speaker_id: スピーカーID  
- speed: 話速  
- pitch: 音程  
- intonation: 抑揚  
- volume: 音量  
- post_phoneme_length: 音素後余白  
#### Ollama
- url: APIエンドポイント  
- model: 使用モデル  
- generation: 生成パラメータ  

### 8. Advanced
#### Pattern Scoring
- exact_match: 完全一致スコア  
- partial_match: 部分一致スコア  
- negative_match: ネガティブマッチスコア  
- threshold: 閾値  
#### Response Filtering
- remove_patterns: 削除パターンのリスト  
- replace_patterns: 置換パターン（キー=置換前、値=置換後）