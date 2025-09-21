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
import signal
import time
import readline
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import numpy as np
import pyaudio
import requests
import speech_recognition as sr
import sounddevice as sd
from scipy.io import wavfile


try:
    import torch
    import torch.cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

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
    use_gpu: bool = True
    audio_device_index: Optional[int] = None


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
    use_gpu: bool = True
    gpu_layers: int = -1


@dataclass
class PerformanceConfig:
    cache_size: int
    history_limit: int
    response_min_length: int
    response_max_length: int
    disk_cache_dir: str = "./audio_cache"
    max_workers: int = 4
    stream_chunk_size: int = 20
    enable_streaming: bool = True


@dataclass
class PatternScoring:
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
        use_gpu = vv_data.get("use_gpu", True)
        return VoicevoxConfig(
            url=vv_data.get("url", "http://localhost:50021"),
            speaker_id=vv_data.get("speaker_id", 1),
            speed=vv_data.get("speed", 1.1),
            pitch=vv_data.get("pitch", 0.0),
            intonation=vv_data.get("intonation", 1.0),
            volume=vv_data.get("volume", 1.0),
            post_phoneme_length=vv_data.get("post_phoneme_length", 0.2),
            use_gpu=use_gpu,
            audio_device_index=None
        )
    
    def get_llm_config(self) -> LLMConfig:
        llm_data = self.config.get("system", {}).get("llm", {})
        gen_data = llm_data.get("generation", {})
        perf_data = self.config.get("system", {}).get("performance", {})
        use_gpu = llm_data.get("use_gpu", True)
        gpu_layers = llm_data.get("gpu_layers", -1)
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
            max_retries=perf_data.get("max_retries", 2),
            use_gpu=use_gpu,
            gpu_layers=gpu_layers
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        perf_data = self.config.get("system", {}).get("performance", {})
        workers = perf_data.get("max_workers", 4)
        streaming = perf_data.get("enable_streaming", True)
        chunk_size = perf_data.get("stream_chunk_size", 20)
        return PerformanceConfig(
            cache_size=perf_data.get("cache_size", 50),
            history_limit=perf_data.get("history_limit", 6),
            response_min_length=perf_data.get("response_min_length", 80),
            response_max_length=perf_data.get("response_max_length", 200),
            max_workers=workers,
            enable_streaming=streaming,
            stream_chunk_size=chunk_size
        )
    
    def get_pattern_scoring(self) -> PatternScoring:
        scoring_data = self.config.get("advanced", {}).get("pattern_scoring", {})
        return PatternScoring(
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
        self.cache_dir = str(Path(cache_dir).resolve())
        self.max_entries = max_entries
        self._lock = threading.Lock()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._cleanup()
    
    def get(self, key: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        path = self._get_path(key)
        if not os.path.exists(path):
            return None, None
        
        try:
            with self._lock:
                with np.load(path, allow_pickle=False) as data:
                    rate = int(data['rate'].item() if hasattr(data['rate'], 'item') else data['rate'])
                    audio = data['audio'].copy()
                os.utime(path, None)
            logger.debug(f"キャッシュヒット: {hashlib.md5(key.encode()).hexdigest()}")
            return rate, audio
        except Exception as e:
            logger.exception(f"キャッシュ読み込みエラー: {e}")
            try:
                os.unlink(path)
            except Exception:
                pass
            return None, None

    def put(self, key: str, rate: int, audio: np.ndarray) -> None:
        try:
            self._cleanup_if_needed()
            path = self._get_path(key)
            
            try:
                cache_dir = os.path.dirname(path)
                with self._lock:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.savez_compressed(path, rate=np.array(rate), audio=audio)
                    logger.debug(f"キャッシュ保存完了: {hashlib.md5(key.encode()).hexdigest()}")
            except Exception as e:
                logger.exception(f"キャッシュ保存エラー（直接保存）: {e}")
        except Exception as e:
            logger.exception(f"キャッシュシステムエラー: {e}")
    
    def _get_path(self, key: str) -> str:
        h = hashlib.md5(key.encode()).hexdigest()
        return str(Path(self.cache_dir) / f"{h}.npz")
    
    def _cleanup_if_needed(self) -> None:
        try:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npz')]
            if len(cache_files) > self.max_entries + 10:
                self._cleanup()
        except Exception as e:
            logger.error(f"キャッシュチェックエラー: {e}")
    
    def _cleanup(self) -> None:
        try:
            with self._lock:
                cache_path = Path(self.cache_dir)
                files = [(os.path.getatime(str(f)), str(f)) 
                        for f in cache_path.glob("*.npz")]
                
                if len(files) > self.max_entries:
                    files.sort()
                    for _, f in files[:len(files) - self.max_entries]:
                        os.unlink(f)
                    logger.info(f"{len(files) - self.max_entries}個の古いキャッシュを削除しました")
        except Exception as e:
            logger.error(f"キャッシュクリーンアップエラー: {e}")


class PatternMatcher:
    def __init__(self, response_patterns: Dict, fallback_patterns: Dict, scoring: PatternScoring):
        self.response_patterns = response_patterns
        self.fallback_patterns = fallback_patterns
        self.threshold = scoring.threshold
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
            if best_category[1] >= self.threshold:
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


class TerminalInputHandler:
    def __init__(self):
        self.stop_event = threading.Event()
        self.input_event = threading.Event()
        self.input_thread = None
    
    def wait_for_keypress(self, prompt_message="スペースキーを押してください..."):
        print(f"\n{prompt_message} [Enter]")
        input()
        return True
    
    def start_recording_session(self):
        self.input_event.clear()
        self.input_thread = threading.Thread(target=self._input_worker)
        self.input_thread.daemon = True
        self.input_thread.start()
        print("\n録音中... [Enter] キーを押して録音を終了")
    
    def _input_worker(self):
        input()
        self.input_event.set()
    
    def wait_for_stop(self, timeout=None):
        return self.input_event.wait(timeout)
    
    def stop(self):
        self.stop_event.set()
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(1.0)


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
        self.input_handler = TerminalInputHandler()
    
    def record_until_keypress(self) -> Optional[str]:
        self.input_handler.wait_for_keypress("質問するにはEnterキーを押してください")
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
        
        self.input_handler.start_recording_session()
        frames = []
        recording_stopped = False

        try:
            while not recording_stopped:
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
                if self.input_handler.wait_for_stop(0.01):
                    recording_stopped = True
                    print("\n録音を終了します...")
        except Exception as e:
            logger.exception(f"録音エラー: {e}")
        finally:
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
            logger.exception(f"文字起こしエラー: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        
        return None
    
    def cleanup(self):
        if hasattr(self, 'p'):
            self.p.terminate()
        self.input_handler.stop()


class AudioStreamer:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_event = threading.Event()
        self.audio_thread = None
    
    def start(self):
        self.stop_event.clear()
        self.audio_thread = threading.Thread(target=self._audio_worker)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def _audio_worker(self):
        while not self.stop_event.is_set():
            try:
                rate, audio, callback = self.audio_queue.get(timeout=0.5)
                self.is_playing = True
                
                try:
                    sd.play(audio, int(rate))
                    sd.wait()
                    if callback:
                        callback()
                except Exception as e:
                    logger.exception(f"音声再生エラー: {e}")
                finally:
                    self.is_playing = False
                    self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"音声再生ワーカーエラー: {e}")
                time.sleep(0.5)
    
    def play(self, rate: int, audio: np.ndarray, callback: Callable = None):
        try:
            self.audio_queue.put((rate, audio, callback), timeout=1.0)
            return True
        except queue.Full:
            logger.error("音声キューが満杯です")
            return False
    
    def clear_queue(self):
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
        except Exception:
            pass
    
    def stop(self):
        self.stop_event.set()
        self.clear_queue()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(1.0)


class VoiceSynthesizer:
    def __init__(self, config: VoicevoxConfig, filter_patterns: Dict, cache_dir: str, 
                enable_streaming: bool = True, cache_size: int = 50):
        self.config = config
        self.filter_patterns = filter_patterns
        self.session = requests.Session()
        self.cache = DiskCache(cache_dir, max_entries=cache_size)
        self.is_speaking = False
        self.audio_completed = threading.Event()
        self.audio_completed.set()
        self.remove_patterns = [re.compile(p) for p in filter_patterns.get("remove", [])]
        self.replace_patterns = {k: v for k, v in filter_patterns.get("replace", {}).items()}
        self.enable_streaming = enable_streaming
        
        self.use_gpu = config.use_gpu and HAS_CUDA
        if self.use_gpu:
            logger.info("VOICEVOXでGPU処理を有効化します")

        self.voicevox_params = {
            "speaker": config.speaker_id,
            "enable_interrogative_upspeak": True
        }
        if self.use_gpu:
            self.voicevox_params["enable_gpu"] = True
    
    def filter_text(self, text: str) -> str:
        if not text:
            return text
        for pattern in self.remove_patterns:
            text = pattern.sub('', text)
        for old, new in self.replace_patterns.items():
            text = text.replace(old, new)
        return text

    def _create_query(self, text: str) -> Dict:
        try:
            query_url = f"{self.config.url}/audio_query"
            query_params = {"text": text, "speaker": self.config.speaker_id}
            if self.use_gpu:
                query_params["enable_gpu"] = True
                
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
            return query_data
        except Exception as e:
            logger.exception(f"VOICEVOX クエリ作成エラー: {e}")
            return None

    def _synthesize_with_query(self, query_data: Dict) -> Tuple[Optional[int], Optional[np.ndarray]]:
        try:
            synthesis_url = f"{self.config.url}/synthesis"
            synthesis_params = self.voicevox_params
            
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
            audio = audio.astype(np.float32)
            if len(audio.shape) == 1:
                audio = np.column_stack((audio, audio))
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            fade_length = int(rate * 0.15)
            if len(audio) > fade_length:
                fade_out = np.linspace(1.0, 0.0, fade_length)
                audio[-fade_length:] *= fade_out[:, np.newaxis]
                
            return rate, audio
            
        except Exception as e:
            logger.exception(f"VOICEVOX 音声合成エラー: {e}")
            return None, None

    def synthesize(self, text: str) -> Tuple[Optional[int], Optional[np.ndarray]]:
        rate, audio = self.cache.get(text)
        if rate is not None:
            return rate, audio
        
        try:
            query_data = self._create_query(text)
            if query_data is None:
                return None, None
                
            rate, audio = self._synthesize_with_query(query_data)
            if rate is not None and audio is not None:
                self.cache.put(text, rate, audio)
                
            return rate, audio
            
        except Exception as e:
            logger.exception(f"VOICEVOX 合成エラー: {e}")
            return None, None
    
    def split_text_for_streaming(self, text: str, chunk_size: int = 20) -> List[str]:
        parts = re.split(r'([。、．，!?！？]+)', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(parts), 2):
            part = parts[i]
            punct = parts[i+1] if i+1 < len(parts) else ""
            
            if len(current_chunk) + len(part) + len(punct) <= chunk_size:
                current_chunk += part + punct
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > chunk_size:
                    for j in range(0, len(part), chunk_size):
                        sub_part = part[j:j+chunk_size]
                        if j + chunk_size >= len(part):
                            current_chunk = sub_part + punct
                        else:
                            chunks.append(sub_part)
                else:
                    current_chunk = part + punct
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def synthesize_streaming(self, text: str, chunk_size: int = 20) -> List[Tuple[int, np.ndarray]]:
        chunks = self.split_text_for_streaming(text, chunk_size)
        results = []
        
        for chunk in chunks:
            rate, audio = self.synthesize(chunk)
            if rate is not None and audio is not None:
                results.append((rate, audio))
                
        return results
    
    def close(self):
        try:
            self.session.close()
            logger.debug("VOICEVOXセッションをクローズしました")
        except Exception as e:
            logger.error(f"VOICEVOXセッションクローズエラー: {e}")


class StreamingLLM:
    def __init__(self, config: LLMConfig, character: CharacterConfig, history_limit: int):
        self.config = config
        self.character = character
        self.session = requests.Session()
        self.history = deque(maxlen=history_limit)
        
        self.use_gpu = config.use_gpu and HAS_CUDA
        if self.use_gpu:
            logger.info(f"LLM推論でGPU処理を有効化します (モデル: {config.model})")
    
    def prepare_streaming_request(self, input_text: str, fallback_data: Dict = None) -> Dict:
        self.history.append({"role": "user", "text": input_text})

        prefix = ""
        suffix = ""
        if fallback_data:
            prefix = fallback_data.get("prefix", "")
            suffix = fallback_data.get("suffix", "")

        history_text = "\n".join([
            f"{item['role'].capitalize()}: {item['text']}" 
            for item in list(self.history)[:-1]
        ])
        recent_history = f"{history_text}\nユーザー: {input_text}"

        payload = {
            "model": self.config.model,
            "prompt": f"{self.character.prompt}\n\n【会話履歴】\n{recent_history}\n\n{self.character.name}: {prefix}",
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": self.config.num_predict,
                "repeat_penalty": self.config.repeat_penalty
            }
        }
        
        if self.use_gpu:
            payload["options"]["gpu_layers"] = self.config.gpu_layers
            
        return payload, suffix
    
    def stream_generate(self, input_text: str, fallback_data: Dict = None) -> Tuple[str, List[str]]:
        payload, suffix = self.prepare_streaming_request(input_text, fallback_data)
        url = f"{self.config.url}/api/generate"
        
        full_response = ""
        chunks = []
        
        try:
            response = self.session.post(
                url, 
                json=payload, 
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                line_text = line.decode('utf-8')
                if not line_text.startswith('data: '):
                    continue
                    
                json_str = line_text[6:]
                try:
                    data = json.loads(json_str)
                    if 'response' in data:
                        chunk = data['response']
                        chunks.append(chunk)
                        full_response += chunk
                except json.JSONDecodeError:
                    continue
            
            if suffix and full_response:
                full_response = f"{full_response} {suffix}"
                chunks.append(f" {suffix}")
            
            self.history.append({"role": "assistant", "text": full_response})
            return full_response, chunks
            
        except requests.RequestException as e:
            logger.exception(f"Ollama API ストリーミングエラー: {e}")
            error_msg = "ちっ、接続できないぜ！サーバーの調子を確認してみろよ！"
            self.history.append({"role": "assistant", "text": error_msg})
            return error_msg, [error_msg]
        except Exception as e:
            logger.exception(f"応答生成エラー: {e}")
            error_msg = "なんか変なことが起きたぜ！もう一度試してみろよ！"
            self.history.append({"role": "assistant", "text": error_msg})
            return error_msg, [error_msg]
    
    def generate(self, input_text: str, fallback_data: Dict = None) -> str:
        try:
            self.history.append({"role": "user", "text": input_text})

            prefix = ""
            suffix = ""
            if fallback_data:
                prefix = fallback_data.get("prefix", "")
                suffix = fallback_data.get("suffix", "")

            history_text = "\n".join([
                f"{item['role'].capitalize()}: {item['text']}" 
                for item in list(self.history)[:-1]
            ])
            recent_history = f"{history_text}\nユーザー: {input_text}"

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
            
            if self.use_gpu:
                payload["options"]["gpu_layers"] = self.config.gpu_layers

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
            
            self.history.append({"role": "assistant", "text": response_text})
            return response_text
            
        except requests.RequestException as e:
            logger.exception(f"Ollama API エラー: {e}")
            return "ちっ、接続できないぜ！サーバーの調子を確認してみろよ！"
        except Exception as e:
            logger.exception(f"応答生成エラー: {e}")
            return "なんか変なことが起きたぜ！もう一度試してみろよ！"
    
    def close(self):
        try:
            self.session.close()
            logger.debug("LLMセッションをクローズしました")
        except Exception as e:
            logger.error(f"LLMセッションクローズエラー: {e}")


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
        self._stop_event = threading.Event()
        self.matcher = PatternMatcher(
            self.response_patterns, 
            self.fallback_patterns, 
            self.pattern_scoring
        )
        self.recorder = AudioRecorder(self.audio_config)
        self.voice = VoiceSynthesizer(
            self.voicevox_config, 
            self.filter_patterns,
            self.perf_config.disk_cache_dir,
            enable_streaming=self.perf_config.enable_streaming,
            cache_size=self.perf_config.cache_size
        )
        self.llm = StreamingLLM(
            self.llm_config,
            self.character_config,
            self.perf_config.history_limit
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.perf_config.max_workers)
        self.audio_streamer = AudioStreamer()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        
        Path(self.perf_config.disk_cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"パンプキントーク初期化: モデル={self.llm_config.model}, GPU={HAS_CUDA}")
    
    def _signal_handler(self, sig, frame):
        print("\n終了処理中です。お待ちください...")
        self._stop_event.set()
        
    def run(self):
        self.audio_streamer.start()
        
        print("=== AI-Pumpkin-Talk ===")
        print("俺様、パンプキンの登場だぜ！何か質問があるなら言ってみろよ！")
        print("質問するにはEnterキーを押してから話し、終了時も同様です")
        print("終了するには Ctrl+C を押してください")
        intro_text = "よう、俺様に会いたかったか？なんでも聞いてみろよ！"
        self._synthesize_and_play(intro_text)
        
        try:
            while not self._stop_event.is_set():
                input_text = self.recorder.record_until_keypress()
                if not input_text or self._stop_event.is_set():
                    continue
                
                category = self.matcher.find_matching_category(input_text)
                
                if category:
                    if category.startswith("fallback:"):
                        fallback_data = self.matcher.get_fallback_data(category)
                        print(f"フォールバックパターンから応答を生成: {category}")
                        self._process_fallback_response(input_text, fallback_data)
                    else:
                        template = self.config_loader.get_template(category)
                        if template:
                            print(f"テンプレート({category})から応答を取得")
                            self._synthesize_and_play(template)
                        else:
                            print("AIで応答を生成中...")
                            self._process_streaming_response(input_text)
                else:
                    print("AIで応答を生成中...")
                    self._process_streaming_response(input_text)
        
        except KeyboardInterrupt:
            print("\n=== パンプキントークシステムを終了します ===")
            self._stop_event.set()
        finally:
            self._cleanup()
    
    def _process_fallback_response(self, input_text: str, fallback_data: Dict):
        if self.perf_config.enable_streaming:
            full_response, chunks = self.llm.stream_generate(input_text, fallback_data)
            filtered_response = self.voice.filter_text(full_response)
            print(f"応答: {filtered_response}")
            self._synthesize_and_play(filtered_response)
        else:
            response_text = self.llm.generate(input_text, fallback_data)
            filtered_response = self.voice.filter_text(response_text)
            print(f"応答: {filtered_response}")
            self._synthesize_and_play(filtered_response)
    
    def _process_streaming_response(self, input_text: str):
        if self.perf_config.enable_streaming:
            print("AIで応答をストリーミング生成中...")
            full_response, chunks = self.llm.stream_generate(input_text)
            
            if chunks:
                accumulated = ""
                parts_to_synthesize = []
                for chunk in chunks:
                    accumulated += chunk
                    if re.search(r'[。、．，!?！？]', chunk):
                        parts_to_synthesize.append(accumulated)
                        accumulated = ""
                
                if accumulated:
                    parts_to_synthesize.append(accumulated)
                
                filtered_parts = [self.voice.filter_text(part) for part in parts_to_synthesize]
                print(f"応答: {''.join(filtered_parts)}")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    audio_futures = [executor.submit(self.voice.synthesize, part) for part in filtered_parts]

                    for future in concurrent.futures.as_completed(audio_futures):
                        rate, audio = future.result()
                        if rate is not None and audio is not None:
                            self.audio_streamer.play(rate, audio)
            else:
                print("応答生成に失敗しました。フォールバックメッセージを再生します。")
                fallback_msg = "ごめんな、うまく応答できなかったぜ。もう一度質問してみてくれ。"
                self._synthesize_and_play(fallback_msg)
        else:
            response_text = self.llm.generate(input_text)
            filtered_response = self.voice.filter_text(response_text)
            print(f"応答: {filtered_response}")
            self._synthesize_and_play(filtered_response)
    
    def _synthesize_and_play(self, text: str):
        try:
            print("音声合成中...")
            if self.perf_config.enable_streaming and len(text) > self.perf_config.stream_chunk_size * 2:
                audio_segments = self.voice.synthesize_streaming(text, self.perf_config.stream_chunk_size)
                for rate, audio in audio_segments:
                    if self._stop_event.is_set():
                        break
                    self.audio_streamer.play(rate, audio)
            else:
                rate, audio = self.voice.synthesize(text)
                if rate is not None and audio is not None:
                    self.audio_streamer.play(rate, audio)
        except Exception as e:
            logger.exception(f"音声合成・再生エラー: {e}")
    
    def _cleanup(self):
        self.executor.shutdown(wait=False)
        self.audio_streamer.stop()
        self.voice.close()
        self.llm.close()
        self.recorder.cleanup()
        logger.info("すべてのリソースを解放しました")


if __name__ == "__main__":
    try:
        import time
        if HAS_CUDA:
            print(f"CUDA対応GPUを検出しました: {torch.cuda.get_device_name(0)}")
            print(f"GPU最適化が有効になります")
        else:
            print("CUDA対応GPUが検出されませんでした。CPU処理で実行します。")
        
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