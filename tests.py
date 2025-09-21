import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import math
import time
import argparse
from typing import List, Tuple

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm


def read_ljspeech_meta(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            uid, text = line.split("|", 1)
            uid = uid.strip().strip('"')
            text = text.strip().strip('"')
            if uid:
                rows.append((uid, text))
    return rows


def nanmean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Папка с metadata.csv и wavs/")
    ap.add_argument("-o", "--output", required=True, help="Куда сохранять результаты")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--context-window", type=int, default=0)  # безопаснее 0
    ap.add_argument("--ref-wav", required=True, help="Путь к reference wav (один на все)")
    ap.add_argument("--start-from", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta_path = os.path.join(args.input, "metadata.csv")
    rows = read_ljspeech_meta(meta_path)
    if not rows:
        print(f"[ERR] пустой/нечитаемый {meta_path}")
        return

    os.makedirs(args.output, exist_ok=True)
    out_wavs = os.path.join(args.output, "wavs")
    os.makedirs(out_wavs, exist_ok=True)

    # Поднимаем модель
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    sr = model.sr
    print(f"Model on {device}, sr={sr}")

    # Готовим reference один раз
    model.prepare_conditionals(args.ref_wav, exaggeration=0.5)

    start_idx = max(0, args.start_from)
    end_idx = len(rows) if args.limit is None else min(len(rows), start_idx + args.limit)

    lines = []
    rtf_vals, ttfta_vals, ttftt_vals = [], [], []

    for i in range(start_idx, end_idx):
        uid, text = rows[i]
        text = punc_norm(text)

        # измеряем стену вручную вокруг всего стриминга
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_start = time.perf_counter()

        wav_chunks = []
        ttft_audio_ms = float("nan")
        ttft_token_ms = float("nan")

        try:
            for audio_chunk, metrics in model.generate_stream(
                text=text,
                language_id=args.lang,
                audio_prompt_path=None,            # уже подготовили референс
                exaggeration=0.5,
                cfg_weight=0.5,
                temperature=0.8,
                chunk_size=args.chunk_size,
                context_window=args.context_window,
                print_metrics=False,
            ):
                if math.isnan(ttft_audio_ms) and metrics.latency_to_first_chunk is not None:
                    ttft_audio_ms = metrics.latency_to_first_chunk * 1000.0
                if math.isnan(ttft_token_ms) and metrics.latency_to_first_token is not None:
                    ttft_token_ms = metrics.latency_to_first_token * 1000.0

                wav_chunks.append(audio_chunk)

        except RuntimeError as e:
            print(f"[ERR] {uid}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            lines.append(f"{uid}|nan|nan|nan|runtime_error")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_end = time.perf_counter()

        if not wav_chunks:
            print(f"[WARN] {uid}: empty audio")
            lines.append(f"{uid}|nan|nan|nan|empty")
            continue

        wav = torch.cat(wav_chunks, dim=-1)
        # сохраняем WAV
        ta.save(os.path.join(out_wavs, f"{uid}.wav"), wav.cpu(), sr)

        audio_s = wav.shape[-1] / sr
        wall_s = wall_end - wall_start

        rtf = wall_s / audio_s if audio_s > 0 else float("nan")

        rtf_vals.append(rtf)
        ttfta_vals.append(ttft_audio_ms)
        ttftt_vals.append(ttft_token_ms)

        lines.append(f"{uid}|{ttft_audio_ms:.1f}|{ttft_token_ms:.1f}|{rtf:.4f}")
        print(f"[OK] {uid}: TTFT(audio)={ttft_audio_ms:.1f} ms, TTFT(token)={ttft_token_ms:.1f} ms, RTF={rtf:.4f}, "
              f"dur={audio_s:.2f}s, wall={wall_s:.2f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_ttfta = nanmean(ttfta_vals)
    avg_ttftt = nanmean(ttftt_vals)
    avg_rtf   = nanmean(rtf_vals)

    with open(os.path.join(args.output, "metadata.metrics.csv"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.write(f"avg|{avg_ttfta:.1f}|{avg_ttftt:.1f}|{avg_rtf:.4f}\n")

    print("\nSaved:")
    print(f"  wavs -> {out_wavs}")
    print(f"  metrics -> {os.path.join(args.output, 'metadata.metrics.csv')}")
    print(f"AVG TTFT(audio)={avg_ttfta:.1f} ms, AVG TTFT(token)={avg_ttftt:.1f} ms, AVG RTF={avg_rtf:.4f}")


if __name__ == "__main__":
    main()
