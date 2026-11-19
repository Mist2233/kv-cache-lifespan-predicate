"""OpenOmni（O2S-300K）数据准备：图像下载 + TTS 音频生成 + 会话构造。

背景
----
HuggingFace `Tongyi-ConvAI/OpenOmni` 的 `openomni_stage3-1.json` 中，
有 101,549 条样本同时包含 `image` 与 `audio` 字段（即 O2S-300K 子集）。
但发布仓库只含 JSON 索引：
  * `image` 是绘图来源路径（COCO / LLaVA / SAM / WikiArt 等）；
  * `audio` 是作者机器上的 `audio_llava/*.wav` 文件，未随数据发布。

本脚本的做法（与 OpenOmni 团队一致）：
  1. 只取 **COCO 来源**的样本（image 以 `coco/train2017/` 开头），
     图像可按 URL 单张下载：http://images.cocodataset.org/train2017/<id>.jpg
  2. 用开源 TTS（microsoft/speecht5_tts）**重新合成**问题语音；
  3. 把若干单轮样本串成一个**多轮会话**（每轮：一张图 + 一段语音提问 + 文本回答），
     以便模拟真实的多轮多模态使用。

用法：
  python src/openomni_prep.py --n-sessions 20 --turns 3 --out data/openomni_small
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()

import argparse
import glob
import io
import json
import os
import urllib.request
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(ROOT, "data/OpenOmni/openomni_stage3-1.json")
COCO_URL = "http://images.cocodataset.org/train2017/{}.jpg"


# ------------------------------------------------------------------ 数据筛选
def build_session_specs(n_sessions: int, turns: int, seed: int = 0):
    """从 O2S 子集里挑选 COCO 样本，串成 n_sessions 个多轮会话。"""
    print(f"加载 {JSON_PATH} ...")
    data = json.load(open(JSON_PATH))
    coco = [x for x in data
            if str(x.get("image", "")).startswith("coco/train2017/")
            and x.get("audio") and x.get("conversations")]
    print(f"O2S 中 COCO 来源样本：{len(coco)}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(coco))[: n_sessions * turns]
    specs = []
    for s in range(n_sessions):
        picks = idx[s * turns:(s + 1) * turns]
        if len(picks) < turns:
            break
        t = []
        for i in picks:
            x = coco[i]
            img_rel = str(x["image"])
            coco_id = os.path.splitext(os.path.basename(img_rel))[0]
            human = next((m["value"] for m in x["conversations"] if m["from"] == "human"), "")
            gpt = next((m["value"] for m in x["conversations"] if m["from"] == "gpt"), "")
            # 去掉人类问题里的 <image> 占位符（图像单独提供）
            human = human.replace("<image>", "").strip()
            t.append({"coco_id": coco_id, "question": human, "answer": gpt})
        specs.append({"sid": s, "turns": t})
    return specs


# ------------------------------------------------------------------ 资源准备
def download_image(coco_id: str, path: str) -> bool:
    if os.path.exists(path):
        return True
    try:
        req = urllib.request.Request(COCO_URL.format(coco_id),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        from PIL import Image
        Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=95)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] 图像下载失败 {coco_id}: {e}")
        return False


_TTS = {}


def _tts_models():
    if not _TTS:
        from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                                  SpeechT5Processor)
        _TTS["proc"] = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        _TTS["model"] = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        _TTS["voc"] = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        fs = sorted(glob.glob(os.path.join(ROOT, "data/TTS/spkrec-xvect/*.npy")))
        import torch
        _TTS["spk"] = torch.tensor(np.load(fs[7300])).unsqueeze(0)
    return _TTS


def synthesize(text: str, path: str) -> bool:
    if os.path.exists(path):
        return True
    try:
        import torch
        m = _tts_models()
        inputs = m["proc"](text=text[:600], return_tensors="pt")
        with torch.no_grad():
            speech = m["model"].generate_speech(inputs["input_ids"], m["spk"],
                                                vocoder=m["voc"])
        x = (speech.numpy() * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(x.tobytes())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] TTS 失败: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--out", default="data/openomni_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--think-mean", type=float, default=6.0)
    ap.add_argument("--arrival-span", type=float, default=20.0)
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out)
    img_dir = os.path.join(out, "images")
    aud_dir = os.path.join(out, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(aud_dir, exist_ok=True)

    specs = build_session_specs(a.n_sessions, a.turns, a.seed)
    print(f"构造 {len(specs)} 个会话 × {a.turns} 轮")

    ok = 0
    for s in specs:
        for t, turn in enumerate(s["turns"]):
            img = os.path.join(img_dir, f"{turn['coco_id']}.jpg")
            aud = os.path.join(aud_dir, f"s{s['sid']}t{t}.wav")
            if download_image(turn["coco_id"], img) and synthesize(turn["question"], aud):
                turn["image"] = img
                turn["audio"] = aud
                ok += 1
    print(f"资源准备完成：{ok}/{len(specs)*a.turns} 轮可用")

    # 补齐调度信息（到达时刻、思考时间）
    rng = np.random.default_rng(a.seed)
    valid = []
    for s in specs:
        ts = [t for t in s["turns"] if t.get("image") and t.get("audio")]
        if len(ts) < 2:
            continue
        s["turns"] = ts
        s["arrive_at"] = float(rng.uniform(0, a.arrival_span))
        s["think_times"] = list(rng.exponential(a.think_mean, len(ts) - 1))
        valid.append(s)
    with open(os.path.join(out, "sessions.json"), "w") as f:
        json.dump(valid, f, ensure_ascii=False, indent=1)
    print(f"写盘：{out}/sessions.json（{len(valid)} 个会话）")


if __name__ == "__main__":
    main()
