#!/usr/bin/env python3
import sys
import requests

BASE = "http://127.0.0.1:50021"
OUT = "voicevox_test_py.wav"
TEXT = "Python からVOICEVOX APIのテストを行っています。"

# プロキシを無効化
NO_PROXY = {"http": None, "https": None}

def check_version():
    r = requests.get(f"{BASE}/version", timeout=5, proxies=NO_PROXY)
    r.raise_for_status()
    print("version:", r.json())

def get_speakers():
    r = requests.get(f"{BASE}/speakers", timeout=5, proxies=NO_PROXY)
    r.raise_for_status()
    return r.json()

def create_audio_query(text, speaker):
    r = requests.post(
        f"{BASE}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=5,
        proxies=NO_PROXY
    )
    r.raise_for_status()
    return r.json()

def synthesis(audio_query, speaker, out_file):
    r = requests.post(
        f"{BASE}/synthesis",
        params={"speaker": speaker},
        headers={"Content-Type": "application/json"},
        json=audio_query,
        timeout=30,
        proxies=NO_PROXY
    )
    r.raise_for_status()
    with open(out_file, "wb") as f:
        f.write(r.content)

def main():
    try:
        check_version()
        speakers = get_speakers()
        print("speakers count:", len(speakers))
        if not speakers:
            print("speakers が空です。VOICEVOX Engine の設定を確認してください。")
            return

        # 最初の speaker の最初の style の id を採る
        try:
            speaker_id = speakers[0]["styles"][0]["id"]
        except Exception as e:
            print("speaker id の抽出に失敗:", e)
            return

        print("使用 speaker_id:", speaker_id)
        audio_query = create_audio_query(TEXT, speaker_id)
        print("audio_query 作成済み")
        synthesis(audio_query, speaker_id, OUT)
        print("synthesis 完了:", OUT)
    except requests.RequestException as e:
        print("HTTP エラー:", e)
    except Exception as e:
        print("その他エラー:", e)

if __name__ == "__main__":
    main()
