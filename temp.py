import os
import re
import io
import json
import wave
import queue
import random
import hashlib
import tempfile
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np
import keyboard
import pyaudio
import requests
import speech_recognition as sr
import sounddevice as sd
from scipy.io import wavfile


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pumpkin_talk.log", encoding="utf-8")
    ]
)
logger = logging.getLogger('pumpkin')


@dataclass
class AudioConfig:
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    chunk_size: int = 1024
    energy_threshold: int = 300
    phrase_threshold: float = 0.3
    pause_threshold: float = 0.8
    non_speaking_duration: float = 0.5
    fade_duration: float = 0.15


@dataclass
class VoicevoxConfig:
    url: str
    speaker_id: int
    speed: float
    pitch: float
    intonation: float
    volume: float
    post_phoneme_length: float


@dataclass
class LLMConfig:
    url: str
    model: str
    fallback_model: str
    temperature: float
    top_p: float
    top_k: int
    num_predict: int
    repeat_penalty: float
    timeout: int
    max_retries: int


@dataclass
class PerformanceConfig:
    cache_size: int
    history_limit: int
    response_min_length: int
    response_max_length: int
    disk_cache_dir: str = "./audio_cache"


@dataclass
class PatternScoring:
    exact_match: int
    partial_match: int
    negative_match: int
    threshold: int


@dataclass
class CharacterConfig:
    name: str
    prompt: str
    mood: str


class ConfigLoader:
    def __init__(self, config_path: str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info("設定ファイルを読み込みました")
            self.used_templates = {}
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            raise
    
    def get_character_config(self) -> CharacterConfig:
        char_data = self.config.get("character", {})
        return CharacterConfig(
            name=char_data.get("name", "パンプキン"),
            prompt=char_data.get("prompt", ""),
            mood=char_data.get("personality", {}).get("base_mood", "arrogant")
        )
    
    def get_audio_config(self) -> AudioConfig:
        audio_data = self.config.get("system", {}).get("audio", {})
        return AudioConfig(
            rate=audio_data.get("sample_rate", 16000),
            channels=audio_data.get("channels", 1),
            chunk_size=audio_data.get("chunk_size", 1024),
            energy_threshold=audio_data.get("energy_threshold", 300),
            phrase_threshold=audio_data.get("phrase_threshold", 0.3),
            pause_threshold=audio_data.get("pause_threshold", 0.8),
            non_speaking_duration=audio_data.get("non_speaking_duration", 0.5)
        )
    
    def get_voicevox_config(self) -> VoicevoxConfig:
        vv_data = self.config.get("system", {}).get("voicevox", {})
        return VoicevoxConfig(
            url=vv_data.get("url", "http://localhost:50021"),
            speaker_id=vv_data.get("speaker_id", 1),
            speed=vv_data.get("speed", 1.1),
            pitch=vv_data.get("pitch", 0.0),
            intonation=vv_data.get("intonation", 1.0),
            volume=vv_data.get("volume", 1.0),
            post_phoneme_length=vv_data.get("post_phoneme_length", 0.2)
        )
    
    def get_llm_config(self) -> LLMConfig:
        llm_data = self.config.get("system", {}).get("llm", {})
        gen_data = llm_data.get("generation", {})
        perf_data = self.config.get("system", {}).get("performance", {})
        return LLMConfig(
            url=llm_data.get("url", "http://localhost:11434"),
            model=llm_data.get("model", "qwen2.5:1.5b"),
            fallback_model=llm_data.get("fallback_model", "gemma2:2b-instruct-jp"),
            temperature=gen_data.get("temperature", 0.6),
            top_p=gen_data.get("top_p", 0.9),
            top_k=gen_data.get("top_k", 40),
            num_predict=gen_data.get("num_predict", 60),
            repeat_penalty=gen_data.get("repeat_penalty", 1.1),
            timeout=perf_data.get("timeout", 20),
            max_retries=perf_data.get("max_retries", 2)
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        perf_data = self.config.get("system", {}).get("performance", {})
        return PerformanceConfig(
            cache_size=perf_data.get("cache_size", 50),
            history_limit=perf_data.get("history_limit", 6),
            response_min_length=perf_data.get("response_min_length", 80),
            response_max_length=perf_data.get("response_max_length", 200)
        )
    
    def get_pattern_scoring(self) -> PatternScoring:
        scoring_data = self.config.get("advanced", {}).get("pattern_scoring", {})
        return PatternScoring(
            exact_match=scoring_data.get("exact_match", 3),
            partial_match=scoring_data.get("partial_match", 1),
            negative_match=scoring_data.get("negative_match", -2),
            threshold=scoring_data.get("threshold", 1)
        )
    
    def get_response_patterns(self) -> Dict:
        return self.config.get("response_templates", {})
    
    def get_fallback_patterns(self) -> Dict:
        return self.config.get("fallback_patterns", {})
    
    def get_filter_patterns(self) -> Dict:
        filter_data = self.config.get("advanced", {}).get("response_filtering", {})
        return {
            "remove": filter_data.get("remove_patterns", []),
            "replace": filter_data.get("replace_patterns", {})
        }
    
    def get_template(self, category: str) -> Optional[str]:
        template_data = self.config.get("response_templates", {}).get(category, {})
        templates = template_data.get("responses", [])
        
        if not templates:
            return None
        
        if category not in self.used_templates:
            self.used_templates[category] = set()
            
        available = [t for t in templates if t not in self.used_templates[category]]
        
        if not available:
            self.used_templates[category].clear()
            available = templates
        
        template = random.choice(available)
        self.used_templates[category].add(template)
        return template


class DiskCache:
    def __init__(self, cache_dir: str, max_entries: int = 100):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        os.makedirs(cache_dir, exist_ok=True)
        self._cleanup()
    
    def get(self, key: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                data = np.load(path)
                os.utime(path, None)
                logger.debug(f"キャッシュヒット: {key[:20]}...")
                return data['rate'], data['audio']
            except Exception as e:
                logger.debug(f"キャッシュ読み込みエラー: {e}")
                os.unlink(path)
        return None, None
    
    def put(self, key: str, rate: int, audio: np.ndarray) -> None:
        self._cleanup()
        path = self._get_path(key)
        try:
            np.savez(path, rate=rate, audio=audio)
            logger.debug(f"キャッシュ保存: {key[:20]}...")
        except Exception as e:
            logger.debug(f"キャッシュ保存エラー: {e}")
    
    def _get_path(self, key: str) -> str:
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npz")
    
    def _cleanup(self) -> None:
        try:
            files = [(os.path.getatime(f), f) for f in 
                    [os.path.join(self.cache_dir, x) for x in os.listdir(self.cache_dir)]
                    if f.endswith('.npz')]
            
            if len(files) > self.max_entries:
                files.sort()
                for _, f in files[:len(files) - self.max_entries]:
                    os.unlink(f)
                    logger.debug(f"古いキャッシュを削除: {os.path.basename(f)}")
        except Exception as e:
            logger.debug(f"キャッシュクリーンアップエラー: {e}")


class PatternMatcher:
    def __init__(self, response_patterns: Dict, fallback_patterns: Dict, scoring: PatternScoring):
        self.response_patterns = response_patterns
        self.fallback_patterns = fallback_patterns
        self.scoring = scoring
        self.compiled_patterns = {}
        self.category_priorities = {}

        for category, data in response_patterns.items():
            if "patterns" in data:
                self.compiled_patterns[category] = [
                    re.compile(p, re.I) for p in data["patterns"]
                ]
                self.category_priorities[category] = data.get("priority", 5)
        self.compiled_fallbacks = {}
        for category, data in fallback_patterns.items():
            if "patterns" in data:
                self.compiled_fallbacks[category] = [
                    re.compile(p, re.I) for p in data["patterns"]
                ]
    
    def find_matching_category(self, query: str) -> Optional[str]:
        if not query:
            return None
        scores = {}
        for category, patterns in self.compiled_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(query):
                    match_count += 1
            if match_count > 0:
                priority = self.category_priorities.get(category, 5)
                scores[category] = match_count * priority
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            if best_category[1] >= self.scoring.threshold:
                logger.info(f"マッチングカテゴリ: '{best_category[0]}' (スコア: {best_category[1]})")
                return best_category[0]
        for category, patterns in self.compiled_fallbacks.items():
            for pattern in patterns:
                if pattern.search(query):
                    logger.info(f"フォールバックマッチ: '{category}'")
                    return f"fallback:{category}"
        return None
    
    def get_fallback_data(self, category: str) -> Dict:
        if category.startswith("fallback:"):
            fb_category = category.split(":", 1)[1]
            return self.fallback_patterns.get(fb_category, {})
        return {}


class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = config.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = config.pause_threshold
        self.recognizer.phrase_threshold = config.phrase_threshold
        self.recognizer.non_speaking_duration = config.non_speaking_duration
        self.p = pyaudio.PyAudio()
    
    def record_until_keypress(self) -> Optional[str]:
        print("\n[SPACE]キーを押して質問を開始...")
        keyboard.wait('space')
        print("\nキーが押されました。質問をどうぞ...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        wave_file = wave.open(tmp_path, 'wb')
        wave_file.setnchannels(self.config.channels)
        wave_file.setsampwidth(self.p.get_sample_size(self.config.format))
        wave_file.setframerate(self.config.rate)
        
        stream = self.p.open(
            format=self.config.format,
            channels=self.config.channels,
            rate=self.config.rate,
            input=True,
            frames_per_buffer=self.config.chunk_size
        )
        print("録音中... [SPACE]キーを押して録音を終了")
        frames = []
        recording_stopped = False

        def on_key_press(e):
            nonlocal recording_stopped
            if e.name == 'space':
                recording_stopped = True
                print("\n録音を終了します...")
        keyboard.on_press(on_key_press)
        
        try:
            while not recording_stopped:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            logger.error(f"録音エラー: {e}")
        finally:
            keyboard.unhook_all()
            stream.stop_stream()
            stream.close()
            for frame in frames:
                wave_file.writeframes(frame)
            wave_file.close()
        try:
            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language="ja-JP")
                print(f"認識されたテキスト: {text}")
                return text
        except sr.UnknownValueError:
            print("音声を認識できませんでした")
        except sr.RequestError as e:
            print(f"音声認識サービスエラー: {e}")
        except Exception as e:
            logger.error(f"文字起こしエラー: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return None
    
    def cleanup(self):
        if hasattr(self, 'p'):
            self.p.terminate()


class VoiceSynthesizer:
    def __init__(self, config: VoicevoxConfig, filter_patterns: Dict, cache_dir: str):
        self.config = config
        self.filter_patterns = filter_patterns
        self.session = requests.Session()
        self.cache = DiskCache(cache_dir)
        self.is_speaking = False
        self.audio_completed = threading.Event()
        self.audio_completed.set()
        self.remove_patterns = [re.compile(p) for p in filter_patterns.get("remove", [])]
        self.replace_patterns = {k: v for k, v in filter_patterns.get("replace", {}).items()}
    
    def filter_text(self, text: str) -> str:
        if not text:
            return text
        for pattern in self.remove_patterns:
            text = pattern.sub('', text)
        for old, new in self.replace_patterns.items():
            text = text.replace(old, new)
        return text
    
    def synthesize(self, text: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        rate, audio = self.cache.get(text)
        if rate is not None:
            return rate, audio
        
        try:
            query_url = f"{self.config.url}/audio_query"
            query_params = {"text": text, "speaker": self.config.speaker_id}
            query_response = self.session.post(query_url, params=query_params, timeout=10.0)
            query_response.raise_for_status()
            query_data = query_response.json()
            query_data.update({
                "speedScale": self.config.speed,
                "pitchScale": self.config.pitch,
                "intonationScale": self.config.intonation,
                "volumeScale": self.config.volume,
                "outputSamplingRate": 24000,
                "outputStereo": False,
                "prePhonemeLength": 0.1,
                "postPhonemeLength": self.config.post_phoneme_length
            })
            synthesis_url = f"{self.config.url}/synthesis"
            synthesis_params = {"speaker": self.config.speaker_id}
            synthesis_response = self.session.post(
                synthesis_url, 
                params=synthesis_params,
                json=query_data,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            synthesis_response.raise_for_status()
            wav_data = io.BytesIO(synthesis_response.content)
            wav_data.seek(0)
            rate, audio = wavfile.read(wav_data)
            if len(audio.shape) == 1:
                audio = np.column_stack((audio, audio))
            fade_length = int(rate * 0.15)
            if len(audio) > fade_length:
                fade_out = np.linspace(1.0, 0.0, fade_length)
                audio[-fade_length:] = audio[-fade_length:] * fade_out[:, np.newaxis]
            self.cache.put(text, rate, audio)
            return rate, audio
        except requests.RequestException as e:
            logger.error(f"VOICEVOX API エラー: {e}")
            print(f"音声合成エラー: {e}")
        except Exception as e:
            logger.error(f"音声合成処理エラー: {e}")
        return None, None
    
    def play(self, rate: Optional[int], audio: Optional[np.ndarray]) -> None:
        if rate is None or audio is None:
            self.audio_completed.set()
            return
        
        try:
            self.is_speaking = True
            self.audio_completed.clear()
            def playback_finished():
                self.is_speaking = False
                print("----- 音声再生完了 -----")
                self.audio_completed.set()

            def monitor_playback():
                sd.play(audio, rate)
                sd.wait()
                playback_finished()
            threading.Thread(target=monitor_playback, daemon=True).start()
            
        except Exception as e:
            logger.error(f"音声再生エラー: {e}")
            self.is_speaking = False
            self.audio_completed.set()


class ResponseGenerator:
    def __init__(self, config: LLMConfig, character: CharacterConfig, history_limit: int):
        self.config = config
        self.character = character
        self.session = requests.Session()
        self.history = deque(maxlen=history_limit)
    
    def generate(self, input_text: str, fallback_data: Dict = None) -> str:
        try:
            self.history.append(f"ユーザー: {input_text}")
            prefix = ""
            suffix = ""
            if fallback_data:
                prefix = fallback_data.get("prefix", "")
                suffix = fallback_data.get("suffix", "")
            recent_history = "\n".join(list(self.history)[-self.history.maxlen:])

            url = f"{self.config.url}/api/generate"
            payload = {
                "model": self.config.model,
                "prompt": f"{self.character.prompt}\n\n【会話履歴】\n{recent_history}\n\n{self.character.name}: {prefix}",
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": self.config.num_predict,
                    "repeat_penalty": self.config.repeat_penalty
                }
            }

            response_text = ""
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = self.session.post(
                        url, 
                        json=payload, 
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get("response", "")
                    if not response_text or len(response_text) < 10:
                        if attempt < self.config.max_retries:
                            logger.warning(f"応答が短すぎます。フォールバックモデルを試行します ({attempt+1}/{self.config.max_retries})")
                            payload["model"] = self.config.fallback_model
                            continue
                    break
                except Exception as e:
                    logger.error(f"LLM API エラー (試行 {attempt+1}): {e}")
                    if attempt < self.config.max_retries:
                        payload["model"] = self.config.fallback_model
                        continue
                    else:
                        raise
            if suffix and response_text:
                response_text = f"{response_text} {suffix}"
            self.history.append(f"{self.character.name}: {response_text}")
            return response_text
            
        except requests.RequestException as e:
            logger.error(f"Ollama API エラー: {e}")
            return "Ollamaに接続できません。"
        except Exception as e:
            logger.error(f"応答生成エラー: {e}")
            return "回答が生成できません。"


class PumpkinTalk:
    def __init__(self, config_file: str = "pumpkin_config.json"):
        self.config_loader = ConfigLoader(config_file)
        self.character_config = self.config_loader.get_character_config()
        self.audio_config = self.config_loader.get_audio_config()
        self.voicevox_config = self.config_loader.get_voicevox_config()
        self.llm_config = self.config_loader.get_llm_config()
        self.perf_config = self.config_loader.get_performance_config()
        self.pattern_scoring = self.config_loader.get_pattern_scoring()
        self.response_patterns = self.config_loader.get_response_patterns()
        self.fallback_patterns = self.config_loader.get_fallback_patterns()
        self.filter_patterns = self.config_loader.get_filter_patterns()
        self.matcher = PatternMatcher(
            self.response_patterns, 
            self.fallback_patterns, 
            self.pattern_scoring
        )
        self.recorder = AudioRecorder(self.audio_config)
        self.voice = VoiceSynthesizer(
            self.voicevox_config, 
            self.filter_patterns,
            self.perf_config.disk_cache_dir
        )
        self.generator = ResponseGenerator(
            self.llm_config,
            self.character_config,
            self.perf_config.history_limit
        )
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_queue = queue.Queue()
        os.makedirs(self.perf_config.disk_cache_dir, exist_ok=True)
        logger.info(f"パンプキントーク初期化: モデル={self.llm_config.model}, スピーカー={self.voicevox_config.speaker_id}")
    
    def run(self):
        print("=== パンプキントーク v2.0.0 ===")
        print("俺様、パンプキンの登場だぜ！何か質問があるなら言ってみろよ！")
        print("質問するには[SPACE]キーを押してから話し、終了時も[SPACE]キーを押してください")
        print("終了するには Ctrl+C を押してください")
        audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        audio_thread.start()
        try:
            while True:
                if not self.voice.audio_completed.is_set():
                    self.voice.audio_completed.wait()
                input_text = self.recorder.record_until_keypress()
                if not input_text:
                    continue
                category = self.matcher.find_matching_category(input_text)
                if category:
                    if category.startswith("fallback:"):
                        fallback_data = self.matcher.get_fallback_data(category)
                        print(f"フォールバックパターンから応答を生成: {category}")
                        response_text = self.generator.generate(input_text, fallback_data)
                    else:
                        template = self.config_loader.get_template(category)
                        if template:
                            print(f"テンプレート({category})から応答を取得")
                            response_text = template
                        else:
                            print("AIで応答を生成中...")
                            response_text = self.generator.generate(input_text)
                else:
                    print("AIで応答を生成中...")
                    response_text = self.generator.generate(input_text)
                response_text = self.voice.filter_text(response_text)
                print(f"応答: {response_text}")
                self.executor.submit(self._synthesize_and_queue, response_text)
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")
        finally:
            self.executor.shutdown(wait=False)
            self.recorder.cleanup()
    
    def _synthesize_and_queue(self, text: str):
        print("音声合成中...")
        rate, audio = self.voice.synthesize(text)
        if rate is not None and audio is not None:
            self.audio_queue.put((rate, audio))
    
    def _audio_worker(self):
        while True:
            try:
                rate, audio = self.audio_queue.get(timeout=1)
                print("再生中...")
                self.voice.play(rate, audio)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"音声再生ワーカーエラー: {e}")


if __name__ == "__main__":
    try:
        app = PumpkinTalk()
        app.run()
    except FileNotFoundError:
        print("エラー: 設定ファイル 'pumpkin_config.json' が見つかりません。")
    except json.JSONDecodeError:
        print("エラー: 設定ファイルの形式が正しくありません。")
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()