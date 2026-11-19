"""生成 3000 组唯一的 (COCO 图, TTS 语音) 素材，降低前缀复用。"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _setup import add_src_paths  # noqa: E402
add_src_paths()
import json, os, sys, io, wave
import numpy as np
import os as _os
_os.environ.setdefault('OMP_NUM_THREADS','2')
_os.environ.setdefault('MKL_NUM_THREADS','2')
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_PATH = os.path.join(ROOT, "data/OpenOmni/openomni_stage3-1.json")
OUT = os.path.join(ROOT, "data/openomni_pool")
IMG_DIR = os.path.join(OUT, "images")
AUD_DIR = os.path.join(OUT, "audio")

def tts_worker(args):
    text, path = args
    if os.path.exists(path):
        return True
    try:
        import torch, glob
        torch.set_num_threads(2)
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        global _M
        try:
            _M
        except NameError:
            proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            fs = sorted(glob.glob(os.path.join(ROOT, "data/TTS/spkrec-xvect/*.npy")))
            spk = torch.tensor(np.load(fs[7300])).unsqueeze(0)
            _M = (proc, model, voc, spk)
        proc, model, voc, spk = _M
        inputs = proc(text=text[:600], return_tensors="pt")
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], spk, vocoder=voc)
        x = (speech.numpy() * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(x.tobytes())
        return True
    except Exception as e:
        print("tts err", e); return False

def dl_worker(item):
    coco_id, path = item
    if os.path.exists(path):
        return True
    try:
        import urllib.request
        from PIL import Image
        req = urllib.request.Request(
            f"http://images.cocodataset.org/train2017/{coco_id}.jpg",
            headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        Image.open(io.BytesIO(data)).convert("RGB").save(path, "JPEG", quality=95)
        return True
    except Exception:
        return False

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    os.makedirs(IMG_DIR, exist_ok=True); os.makedirs(AUD_DIR, exist_ok=True)
    data = json.load(open(JSON_PATH))
    seen, picks = set(), []
    for x in data:
        img = str(x.get("image", ""))
        if not img.startswith("coco/train2017/") or not x.get("audio") or not x.get("conversations"):
            continue
        cid = os.path.splitext(os.path.basename(img))[0]
        if cid in seen:
            continue
        seen.add(cid)
        q = next((m["value"] for m in x["conversations"] if m["from"] == "human"), "")
        a = next((m["value"] for m in x["conversations"] if m["from"] == "gpt"), "")
        picks.append({"coco_id": cid, "question": q.replace("<image>", "").strip(), "answer": a})
        if len(picks) >= n:
            break
    print(f"选中 {len(picks)} 组唯一素材")

    imgs = [(p["coco_id"], os.path.join(IMG_DIR, p["coco_id"] + ".jpg")) for p in picks]
    with ThreadPoolExecutor(16) as ex:
        ok_img = sum(ex.map(dl_worker, imgs))
    print(f"图像下载完成 {ok_img}/{len(imgs)}")

    tasks = [(p["question"], os.path.join(AUD_DIR, f"{i}.wav")) for i, p in enumerate(picks)]
    with ProcessPoolExecutor(12) as ex:
        ok_aud = sum(ex.map(tts_worker, tasks, chunksize=8))
    print(f"语音合成完成 {ok_aud}/{len(tasks)}")

    pool = []
    for i, p in enumerate(picks):
        ip, ap = imgs[i][1], tasks[i][1]
        if os.path.exists(ip) and os.path.exists(ap):
            pool.append({"image": ip, "audio": ap, "question": p["question"], "answer": p["answer"]})
    json.dump(pool, open(os.path.join(OUT, "pool.json"), "w"))
    print(f"素材池就绪：{len(pool)} 组 -> {OUT}/pool.json")

if __name__ == "__main__":
    main()
