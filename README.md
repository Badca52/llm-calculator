# LLM VRAM Calculator

Desktop tool that takes a Hugging Face model URL and shows VRAM requirements
across quantization formats and context lengths, with live recomputation as you
adjust a GPU-offload slider.

## Requirements

- Python 3.11+
- PySide6 6.7+ (installed from `requirements.txt`)

## Install & run

```bash
git clone https://github.com/Badca52/llm-calculator.git
cd llm-calculator
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

## Usage

1. Paste a Hugging Face model URL (e.g.
   `https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`).
2. For gated models (Llama, Gemma, etc.), paste an HF access token in the
   token field. The token is held in memory only — it is never written to
   disk or logged.
3. Adjust the VRAM budget, KV cache dtype, and GPU-layers slider. The matrix
   recomputes automatically.
4. Cell colors: **green** = fits comfortably, **yellow** = tight (<1 GB
   headroom), **red** = exceeds budget. Bold row labels mark quantizations
   actually available as `.gguf` files in the repo.

### Shortcuts

| Key      | Action                |
|----------|-----------------------|
| `Ctrl+T` | Toggle dark / light   |
| `Ctrl+Q` | Quit                  |

## Methodology

VRAM estimates use the standard three-term decomposition:

```
weights + KV cache + runtime overhead (~300 MB)
```

Embedding and `lm_head` always count on GPU when any transformer layer is
offloaded (matches llama.cpp behavior). Effective bits-per-weight values for
GGUF quantizations come from measured real-world file sizes — see
`quantization.py` for the full table.

Estimates are accurate to within ±3–5%. Actual VRAM varies with the inference
backend, batch size, and platform.

## Files

| File              | Purpose                                   |
|-------------------|-------------------------------------------|
| `main.py`         | Entry point, `MainWindow`, UI wiring      |
| `calculator.py`   | Pure VRAM math                            |
| `hf_client.py`    | Hugging Face API + `QThread` worker       |
| `quantization.py` | BPW tables and context-size helpers       |

## License

MIT — see [`LICENSE`](LICENSE).
