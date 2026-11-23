"""A native voice for one language of the app, from a clean recording of about 8 s (the Arabic and Japanese voices; the
Russian one was made the same way, pc/export_ru_voice.py). Writes:
  chat/assets/t3_prefix.bin        the <lang> block <- the 34 prefix rows of T3 conditioned on the recording (added if
                                   the file has no block for <lang> yet; the other blocks unchanged). With --rows
                                   speaker, row 0 (the speaker row) stays Garlic's own and rows 1-33 come from the clip.
  chat/assets/s3g_voice_<lang>.bin S3Gen's voice for <lang>, as pc/export_s3g_assets.py lays it out: prompt tokens,
                                   prompt mel [2P][80], spk = affine(normalize(x-vector)) of the meanflow decoder, then
                                   the gain for its audio (f32), so it sounds as loud as Garlic's own voice
The conditionals (Chatterbox's prepare_conditionals, exaggeration 0.5) are saved next to the recording, as
<recording>.conds.pt. The recording's prompt must stay within 216 speech tokens (8.6 s): S3Gen's work area on the
console holds the prompt and the sentence together.

The voices of the app (Mozilla Common Voice 17, CC0; each recording is a few clips of one speaker of the train set,
trimmed at both ends, levelled to -21 dBFS of speech and joined with 0.08 s of silence, 24 kHz). Each clip is cleaned
with Resemble Enhance (MIT; nfe 128, midpoint solver, lambd 0.9, tau 0.5, as the Russian voice) and then cut at the
same samples, so the recording keeps its length:
  ar   8.7 s: common_voice_ar_24161342, _24162090, _24166017, _24157386     --rows speaker --gain-db 3.8
  ja   7.1 s: common_voice_ja_19682143, _19682147, _19682962                --rows speaker --gain-db 2.2
Both keep Garlic's speaker row: with Whisper large-v3 as the judge (25 sentences per language, 2 seeds), it gave the
fewest misheard characters, and Whisper hears the language more surely than with Garlic's own voice. The cleaned
recordings are understood as well as the raw ones (120 sentence pairs per language) and give quieter backgrounds
(Arabic +6 dB, Japanese +12 dB). The gains bring them to Garlic's loudness (Arabic about 1 dB under it, to keep the
limiter from working all the time; its cleaned recording comes out 1.7 dB louder than the raw one, hence 3.8).

Run: python pc/export_lang_voice.py <lang> <recording.wav> [--rows all|speaker] [--gain-db G]
     (needs torch and the chatterbox package, and models/chatterbox/hf: ve, t3_mtl23ls_v3 and s3gen safetensors)
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))
CB = ROOT / "models" / "chatterbox"
HF = CB / "hf"
ASSETS = ROOT / "chat" / "assets"
MAX_PROMPT = 216


def conditionals(wav, out):
    import torch
    from safetensors.torch import load_file
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
    from chatterbox.models.s3gen import S3Gen
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.models.voice_encoder import VoiceEncoder
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(HF / "ve.safetensors"))
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(HF / "t3_mtl23ls_v3.safetensors"))
    s3 = S3Gen()
    s3.load_state_dict(load_file(HF / "s3gen.safetensors"), strict=False)
    m = ChatterboxMultilingualTTS(t3, s3, ve.eval(), MTLTokenizer(str(HF / "grapheme_mtl_merged_expanded_v1.json")), "cpu")
    m.t3.eval(), m.s3gen.eval()
    m.prepare_conditionals(str(wav), exaggeration=0.5)
    m.conds.save(out)
    with torch.inference_mode():
        return t3.prepare_conditioning(Conditionals.load(out).t3)[0].half().numpy()


def voice_bin(conds_path, P, gain_db):
    import s3gen_ref as R
    v = R.load_voice_conds(conds_path, P=P)
    tok, mel, spk = v["prompt_tok"].astype(np.int32), v["prompt_mel"].astype(np.float32), v["spk"].astype(np.float32)
    assert len(tok) <= MAX_PROMPT + 1, f"{len(tok)} prompt tokens: the recording is too long"
    tok = tok[:MAX_PROMPT]
    assert 2 * len(tok) <= mel.shape[0] <= 2 * len(tok) + 3, mel.shape
    mel = np.ascontiguousarray(mel[: 2 * len(tok)])   # exactly 2 frames per prompt token, as the console reads it
    return (struct.pack("<4sII", b"PS3V", len(tok), mel.shape[0]) + tok.tobytes() + mel.tobytes() + spk.tobytes() +
            struct.pack("<f", 10 ** (gain_db / 20)))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("lang")
    a.add_argument("wav", type=Path)
    a.add_argument("--rows", choices=["all", "speaker"], default="all")
    a.add_argument("--gain-db", type=float, default=0.0)
    args = a.parse_args()
    import s3gen_ref as R
    P = R.prepare(R.load_raw())
    cp = args.wav.with_suffix(".conds.pt")
    prefix = conditionals(args.wav, cp)
    d = bytearray((ASSETS / "t3_prefix.bin").read_bytes())
    assert d[:4] == b"PT3P"
    n = struct.unpack("<I", d[4:8])[0]
    langs = [bytes(d[8 + 4 * i: 12 + 4 * i]).rstrip(b"\0").decode() for i in range(n)]
    blocks = [bytes(d[8 + 4 * n + i * 34 * 2048: 8 + 4 * n + (i + 1) * 34 * 2048]) for i in range(n)]
    if args.rows == "speaker":   # Garlic's speaker row (block 0, the built-in voice)
        prefix = prefix.copy()
        prefix[0] = np.frombuffer(blocks[0], np.float16).reshape(34, 1024)[0]
    block = np.ascontiguousarray(prefix.astype(np.float16)).tobytes()
    assert len(block) == 34 * 2048
    if args.lang in langs:
        blocks[langs.index(args.lang)] = block
    else:
        langs.append(args.lang)
        blocks.append(block)
    out = struct.pack("<4sI", b"PT3P", len(langs)) + b"".join(l.encode().ljust(4, b"\0") for l in langs) + b"".join(blocks)
    (ASSETS / "t3_prefix.bin").write_bytes(out)
    vb = voice_bin(cp, P, args.gain_db)
    (ASSETS / f"s3g_voice_{args.lang}.bin").write_bytes(vb)
    np_ = struct.unpack("<I", vb[4:8])[0]
    print(f"t3_prefix.bin: blocks {langs}; s3g_voice_{args.lang}.bin: {np_} prompt tokens ({np_ / 25:.2f} s), "
          f"gain {args.gain_db:+.1f} dB, {len(vb)} bytes")


if __name__ == "__main__":
    main()
