# Credits and licences

Garlic LLM's own code is licensed under the GNU General Public License v3.0 (see [`LICENSE`](LICENSE)). It stands on the work listed here. Each item keeps its own licence.

## Models

| Model | Author | Licence | Use in Garlic LLM |
|---|---|---|---|
| [Qwen3.5-9B and Qwen3.5-4B](https://huggingface.co/Qwen) | Alibaba Qwen | Apache-2.0 | The chat models. Weights quantized to 4 bits. |
| [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) (T3) | Resemble AI | MIT | Text to speech, first stage. Weights quantized to 4 bits. |
| [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) (meanflow S3Gen and HiFT vocoder) | Resemble AI | MIT | Text to speech, second stage. |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | NVIDIA | CC-BY-4.0 | Speech to text in 25 European languages. Weights quantized to 4 bits. |
| [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | NVIDIA | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) | Speech to text in Arabic, Japanese, Korean, Chinese, Turkish and Vietnamese. Weights quantized to 4 bits, the joint network to 8 bits. The licence text ships inside the app, in [`chat/assets/LICENCIA_NEMOTRON.txt`](chat/assets/LICENCIA_NEMOTRON.txt). |

## Voices

- The default voice is Chatterbox's built-in voice. Its pronunciation in Spanish, Portuguese, Italian, French and German is conditioned on reference clips from Chatterbox's official demo (Resemble AI).
- The Russian voice is cloned from a recording of [Mozilla Common Voice 17](https://commonvoice.mozilla.org) (CC0), cleaned with [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT).
- The Arabic and Japanese voices are cloned from recordings of [Mozilla Common Voice 17](https://commonvoice.mozilla.org) (CC0): a few clips of one speaker each, trimmed, levelled and cleaned with [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) (`pc/export_lang_voice.py`).

## Code inside the app

| Component | Author | Licence |
|---|---|---|
| [MicroPython](https://micropython.org) 1.26.1 (`chat/py/mp/`) | Damien P. George and contributors | MIT (`chat/py/mp/LICENSE`) |
| [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain): C library, start-up code and system modules linked into or packaged with the app | The OpenOrbis team | GPLv3 |

## Data inside the app

| Data | Source | Licence |
|---|---|---|
| Emoji images | [Twemoji](https://github.com/jdecked/twemoji) 16.0.1, Copyright 2019 Twitter, Inc. and other contributors. Images scaled. | CC-BY-4.0 |
| Text fonts, converted to bitmaps | Selawik (Microsoft), Noto Sans, Noto Sans Arabic, Noto Sans Symbols, Noto Sans Symbols 2, Noto Sans Math, Noto Music, Noto Sans SC, Noto Sans TC, Noto Sans JP, Noto Sans KR, Arimo, Inconsolata, Cascadia Mono (Microsoft) | SIL Open Font License 1.1 |
| Symbol font, converted to bitmaps | DejaVu Sans 2.37 | Bitstream Vera licence and public domain |
| Russian stress dictionary (`ru_stress.bin`) | Built with [russian-text-stresser](https://github.com/Vuizur/add-stress-to-epub) by Hannes Krumbiegel (GPLv3), which uses Wiktionary data (CC BY-SA). Word frequencies from [FrequencyWords](https://github.com/hermitdave/FrequencyWords) (OpenSubtitles 2018; CC BY-SA 4.0). | GPLv3 and CC BY-SA |
| Japanese kanji readings (`ja_kana.bin`) | The dictionary of [pykakasi](https://codeberg.org/miurahr/pykakasi) 2.3.0 by Hiroshi Miura and contributors: the KAKASI dictionary (Hironobu Takahashi, Masahiko Sato and others, made from the SKK dictionary) and part of [UniDic](https://clrd.ninjal.ac.jp/unidic/) (The UniDic Consortium). Converted to a compact table (`pc/export_ja_kana.py`). | GPL-3.0-or-later (KAKASI: GPL-2.0-or-later; UniDic: GPL-2.0, LGPL-2.1 or BSD-3-Clause) |
| Language identifier (`chat/lang_id_table.h`) | Character n-gram statistics computed from Wikipedia text | CC BY-SA |
| Tokenizer tables | From the Qwen3.5, Chatterbox, Parakeet and Nemotron 3.5 ASR releases | As their models |

The copyright notices of the fonts, their reserved names and the full text of their licences are in [`chat/assets/LICENCIAS_FUENTES.txt`](chat/assets/LICENCIAS_FUENTES.txt), which ships inside the app together with `chat/assets/CREDITOS.txt` and `chat/assets/LICENCIA_NEMOTRON.txt`.

## Techniques

- The 4-bit attention cache follows *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate* (Zandieh et al., arXiv:2504.19874).
- Speculative decoding uses the multi-token prediction (MTP) head that ships with Qwen3.5.
- Nemotron 3.5 ASR's 4-bit weights are rounded with *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (Frantar et al., arXiv:2210.17323), calibrated on sentences of the dev split of [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google, CC-BY 4.0).

## Tools

Garlic LLM is built with the OpenOrbis PS4 Toolchain and [Zig](https://ziglang.org)'s bundled clang and lld, and runs on consoles with [GoldHEN](https://github.com/GoldHEN/GoldHEN). OpenOrbis and GoldHEN are third-party projects by their own authors; Garlic LLM would not exist without them.

## Logo

The garlic mark, the icon and the home screen background were drawn for Garlic LLM (`pc/make_brand.py`).

---

Not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation and PS4 are trademarks of their owners.
