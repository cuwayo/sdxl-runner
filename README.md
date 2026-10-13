# sdxl-runner

Run the SDXL model **[cagliostrolab/animagine-xl-4.0-zero](https://huggingface.co/cagliostrolab/animagine-xl-4.0-zero)**
on a cloud GPU (Colab, Kaggle, RunPod, Vast, …) and drive it from your own machine.

It ships two entry points:

| Command      | Runs on      | What it does                                                    |
| ------------ | ------------ | --------------------------------------------------------------- |
| `sdxl-server`| the GPU host | Loads Animagine XL 4.0 behind a Gradio API, prints a public URL. |
| `sdxl-runner`| your machine | Submits prompts, downloads the generated images.                 |

There is also a `SdxlClient` Python class for use in scripts and notebooks.

## How it works

```
  LOCAL MACHINE                          CLOUD GPU (Colab / RunPod / ...)
  ┌───────────────────────┐             ┌────────────────────────────────────┐
  │ sdxl-runner (CLI)     │  HTTPS      │ sdxl-server                        │
  │  or SdxlClient (lib)  │ ─────────▶  │  Gradio Blocks (named endpoints)   │
  │  gradio_client.Client │  via        │   /txt2img  /img2img  /loras       │
  │  saves PNGs locally   │  cloudflared│  SDXL + img2img pipelines + LoRA   │
  └───────────────────────┘  tunnel URL └────────────────────────────────────┘
```

The server binds Gradio to `127.0.0.1:PORT`; a free **cloudflared quick tunnel**
exposes it at a public `https://<random>.trycloudflare.com` URL that you paste into
the client.

## Quickstart

There are two ways to run the server on Colab. **Option A** drives Colab entirely
from your terminal with Google's Colab CLI; **Option B** runs it inside a notebook.

### Option A — Colab CLI (terminal only, no browser)

Requires Google's Colab CLI: `pip install google-colab-cli` (and `colab auth login` once).

```bash
# Provision a T4 GPU, install deps, and start the server in the background.
# Prints the public URL and the exact client command to run next.
PKG="sdxl-runner[server] @ git+https://github.com/<you>/sdxl-runner.git" \
  scripts/colab_up.sh
```

Then, on the same machine, run the printed command:

```bash
sdxl-runner generate \
  --url https://xxxx.trycloudflare.com \
  --quality --auto-size \
  -p "1girl, arima kana, oshi no ko, looking at viewer, smile, outdoors, night" \
  -o ./out

colab stop -s sdxl      # release the GPU when you are done
```

`scripts/colab_up.sh` is a thin wrapper around the Colab CLI:

| Step | Command it runs |
| ---- | --------------- |
| Provision GPU | `colab new -s sdxl --gpu T4` |
| Install deps | `colab install -s sdxl diffusers transformers accelerate safetensors peft gradio` + the package |
| Start server | `colab exec -s sdxl --timeout 300 -f scripts/colab_serve.py` |
| Teardown | `colab stop -s sdxl` |

Overridable with env vars: `SESSION` (default `sdxl`), `GPU` (default `T4`;
`GPU=` for CPU), `PORT` (default `8000`), `PKG` (package spec to install).
`scripts/colab_serve.py` launches the server detached so it keeps running after
the cell returns; add `SDXL_MOCK=1` for a no-model wiring test.

### Option B — inside a notebook

```bash
pip install "sdxl-runner[server] @ git+https://github.com/<you>/sdxl-runner.git"
sdxl-server --port 8000
```

It prints a banner with the **public URL**. Or use the ready-made notebook:
[`notebooks/animagine_xl_colab.ipynb`](notebooks/animagine_xl_colab.ipynb).

### 2. On your machine

```bash
pip install "sdxl-runner @ git+https://github.com/<you>/sdxl-runner.git"

sdxl-runner generate \
  --url https://xxxx.trycloudflare.com \
  --quality --auto-size \
  -p "1girl, arima kana, oshi no ko, looking at viewer, smile, outdoors, night" \
  -o ./out
```

## CLI reference

```
sdxl-runner generate --url URL -p PROMPT [options]
sdxl-runner img2img  --url URL -p PROMPT -i INIT.png [--strength 0.6] [options]
sdxl-runner loras    --url URL
```

Common options:

| Flag              | Meaning                                                        |
| ----------------- | -------------------------------------------------------------- |
| `-p, --prompt`    | Positive prompt (required).                                    |
| `-n, --negative`  | Negative prompt (defaults to the model's recommended one).     |
| `-W/-H`           | Width/height (default `832x1216`).                             |
| `--steps`         | Inference steps (default `28`).                                |
| `--cfg`           | Guidance scale (default `5.0`).                                |
| `--seed`          | Seed; `-1` (default) picks a random one.                       |
| `-b, --batch`     | Number of images.                                              |
| `--quality`       | Append the model's quality tags (`masterpiece, high score, …`).|
| `--auto-size`     | Snap `-W/-H` to the nearest recommended resolution bucket.     |
| `--lora`          | Apply a LoRA: `SOURCE[:WEIGHT_NAME][@SCALE]` (repeatable).     |
| `-o, --out`       | Output directory.                                              |
| `--json`          | Print a JSON summary (seed + image paths).                     |

## Python API

```python
from sdxl_runner import SdxlClient, GenerationParams, LoraSpec

client = SdxlClient("https://xxxx.trycloudflare.com", download_dir="./out")

# Convenience form
result = client.generate(
    prompt="1girl, firefly (honkai: star rail), honkai: star rail, safe, outdoors",
    width=832, height=1216, quality_tags=True,
)
print(result.seed, result.images)

# Explicit form, with a LoRA
params = GenerationParams(
    prompt="1girl, solo, looking at viewer",
    loras=[LoraSpec(source="org/lora-repo", weight_name="style.safetensors", scale=0.8)],
)
client.txt2img(params)
```

## Model defaults

Baked in from the model card:

- **Negative prompt**: the model's recommended negative prompt.
- **Quality tags**: `masterpiece, high score, great score, absurdres`.
- **Settings**: Euler a sampler, CFG 5, 28 steps.
- **Resolutions**: the recommended buckets (832×1216, 1024×1024, 1216×832, …).

## Notes & caveats

- **Transport**: Cloudflare quick tunnels carry WebSockets and plain HTTP but **not
  Server-Sent Events (SSE)**. The launcher checks which protocol Gradio negotiated and
  warns if it is SSE. If you hit hangs, restart the server with `--share` (Gradio's own
  `gradio.live` link) or `--no-tunnel` on a host with an open port (RunPod/Vast).
- **First request is slow**: the model (~6.9 GB) loads lazily on first generation.
- **VRAM**: on a 16 GB Colab T4 the server uses `enable_model_cpu_offload()` plus VAE
  slicing/tiling automatically.
- **Testing without a GPU**: `sdxl-server --mock` returns placeholder images so you can
  verify the client ↔ server ↔ tunnel path end-to-end.

## Development

```bash
uv sync --extra server   # or: pip install -e ".[server]"
uv run pytest
```
