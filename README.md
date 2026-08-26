# Qwen3.8 27B FastMTP on Vast.ai

Reproducible, disposable deployment for:

- `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`
- llama.cpp + HauhauCS FastMTP
- one 48 GB NVIDIA GPU
- ~200k–250k context
- Vast.ai billed only while the disposable instance exists/runs
- a local SSH tunnel instead of exposing the inference API directly to the internet

> **v2 cold-start update:** if you used the original version, rebuild/push the new GHCR image before renting another host. Also copy the new `GPU_QUERY` into your local `.env`; `.env` is intentionally gitignored and therefore cannot be updated by `git pull`.

The Docker image contains **no model weights and no credentials**. GitHub Actions builds the CUDA/llama.cpp runtime and publishes it to GHCR. The runtime now derives from Vast.ai's own CUDA base image so SSH-mode hosts can reuse commonly cached/compatible layers instead of spending minutes installing SSH tooling into a generic CUDA image. `qwen-up` rents a suitable Vast host, starts the image, downloads the GGUF from Hugging Face, creates an SSH tunnel, and waits for `/health`. `qwen-down` destroys the Vast instance.

## Defaults

- Model: `Q4_K_P` (17.9 GB)
- Context: `245760`
- FastMTP depth: `3`
- GPU search: RTX 6000 Ada / L40S / RTX 5880 Ada / RTX A6000, 48 GB+, 32 GB system RAM+, CUDA 12.8+, 500 Mbps+ download, 200 MB/s+ disk
- Maximum all-in Vast price: `$0.80/h`
- Local OpenAI-compatible endpoint: `http://127.0.0.1:18080/v1`

Qwen3.8-27B has native 262,144-token context and a hybrid attention layout (16 full-attention layers out of 64), so long context is substantially more memory-friendly than a classic dense full-attention 27B model.

## 1. Create the public GitHub repository

Create an empty public repository, copy these files into it, and push to `main`:

```bash
git init
git add .
git commit -m "Initial Qwen3.8 Vast deployment"
git branch -M main
git remote add origin git@github.com:YOUR_USER/YOUR_REPO.git
git push -u origin main
```

The workflow in `.github/workflows/docker.yml` builds the patched CUDA image and pushes:

```text
ghcr.io/YOUR_USER/YOUR_REPO:latest
ghcr.io/YOUR_USER/YOUR_REPO:sha-...
```

After the first successful build, open the generated GHCR package settings and change package visibility to **Public**. This lets Vast pull the image anonymously. The repository can be public while `.env` remains local and gitignored.

## 2. Hugging Face access

The HauhauCS repository is currently publicly downloadable, so **no Hugging Face token is required** for the default deployment. `hf download` runs anonymously when `HF_TOKEN` is empty.

You may still set a fine-grained/read-only token in local `.env` for authenticated downloads/rate limits or as insurance if the upstream repository's access policy changes later. Never commit it. If present, `start.sh` unsets the token before `exec`-ing `llama-server`.

## 3. Install local dependencies

Linux/macOS/WSL:

```bash
python3 -m pip install --upgrade vastai
sudo apt-get install -y jq curl openssh-client openssl   # Debian/Ubuntu/WSL
```

Authenticate Vast either with:

```bash
vastai set api-key YOUR_VAST_API_KEY
```

or set `VAST_API_KEY` in `.env`.

Also add an SSH public key to your Vast account. `qwen-up` uses Vast's SSH launch mode and an SSH local-forward for the API.

## 4. Configure

```bash
cp .env.example .env
chmod 600 .env
```

Edit at least:

```dotenv
GHCR_IMAGE=ghcr.io/YOUR_USER/YOUR_REPO:latest
```

`HF_TOKEN` is optional.

Recommended starting point:

```dotenv
MODEL=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
CTX_SIZE=245760
MAX_DPH=0.80
```

For more quantization quality, try Q5:

```dotenv
MODEL=Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf
CTX_SIZE=204800
```

Then raise the Q5 context after verifying VRAM headroom on your chosen 48 GB GPU.

## 5. Start

```bash
./qwen-up
```

The script:

1. searches verified/rentable Vast offers matching `GPU_QUERY` (including a minimum disk read bandwidth);
2. rejects anything above `MAX_DPH`;
3. rents the cheapest remaining offer;
4. starts your GHCR image in Vast SSH mode;
5. downloads the target GGUF + ~903 MB FastMTP sidecar with `hf_xet`;
6. starts patched `llama-server` on `127.0.0.1:8080` inside the instance;
7. creates an SSH tunnel to local port `18080`;
8. waits until `/health` responds;
9. writes client environment variables to `.qwen-vast/env`.

Load them:

```bash
source .qwen-vast/env
```

Then test:

```bash
curl "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Example OpenAI-compatible chat call:

```bash
curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
    "messages": [{"role":"user","content":"Review this function for race conditions."}],
    "max_tokens": 1024
  }'
```

## 6. Logs / status

```bash
./qwen-status
./qwen-logs
```

`qwen-logs` tails `/var/log/qwen38/server.log` through SSH.

## 7. Stop billing

```bash
./qwen-down --yes
```

This calls `vastai destroy instance`, not `stop`. Destroying is intentional for this disposable workflow: it removes the container/disk instead of leaving storage charges behind.

If `qwen-up` fails or is interrupted before readiness, it destroys the paid instance automatically unless you explicitly set:

```dotenv
KEEP_ON_FAILURE=1
```

Use that only for debugging.

## Cost controls

`MAX_DPH` is a hard local guard before rental. Example:

```dotenv
MAX_DPH=0.55
```

If no matching host is available below the cap, `qwen-up` exits without renting anything.

`GPU_QUERY` is normal Vast search syntax. Examples:

Fast Ada-only:

```dotenv
GPU_QUERY='num_gpus=1 gpu_name=RTX_6000Ada gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1'
```

Cheapest 48 GB option:

```dotenv
GPU_QUERY='num_gpus=1 gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1'
```

Europe example:

```dotenv
GPU_QUERY='num_gpus=1 gpu_name in ["RTX_6000Ada","L40S","RTX_5880Ada","RTX_A6000"] gpu_ram>=48 cpu_ram>=32 reliability>0.98 inet_down>=500 disk_bw>=200 cuda_vers>=12.8 disk_space>=60 rented=False rentable=True direct_port_count>=1 geolocation in [DE,NL,FR,SE,FI]'
```

## Why the SSH tunnel?

The inference port is deliberately bound to `127.0.0.1` inside the Vast container. Your local client reaches it through SSH port forwarding. This means source code and prompts are encrypted in transit and you do not need to expose a random public HTTP port.

The local generated API key is still enabled as defense in depth.

## Docker build details

The runtime derives from:

```text
vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu24.04-py312
```

Vast recommends extending its pre-built base images for custom images. They are designed for Vast instance setup and their large base layers are commonly cached on hosts, which reduces cold-start overhead compared with a generic CUDA runtime image. The same base is used by both Docker build stages so GitHub Actions does not need two unrelated multi-gigabyte CUDA base-image families.

The image pins llama.cpp to the commit specified by the HauhauCS FastMTP model card:

```text
4df29be4f4c3673f428170fda944a5b19f743bb8
```

At image-build time it downloads the small FastMTP runtime patch from the model repository and verifies the release-manifest SHA-256 before applying it:

```text
981285400b59dc45cf99936b6ff66d4b3aa0f1b532f85fa51418cb407e51d615
```

The model download is also pinned to the release commit `993a5971fda8f30dd1b7eb2654792ba4415c7460` by default (`HF_REVISION`), so a later change on the model repo's `main` branch does not silently change your deployment.

The model weights are intentionally not baked into the image. This keeps your custom GHCR layers small and lets you switch Q4/Q5 without rebuilding.

## Fast download path

The runtime pins `huggingface_hub` with `hf_xet` and sets `HF_XET_HIGH_PERFORMANCE=1`. The default search asks for at least 32 GB system RAM, 500 Mbps download, and 200 MB/s local disk read bandwidth.

The disk filter matters for cold starts: Vast's SSH compatibility/bootstrap steps and model loading can be painfully slow on low-end SATA hosts even when internet bandwidth is good. Raise `disk_bw` to 300–500 if you prefer faster starts over the absolute cheapest hourly offer.

If a host's model download is still poor, increase `inet_down` in `GPU_QUERY` rather than adding persistent Vast storage first. The Q4 target plus FastMTP sidecar is under 20 GB, so a disposable download is usually economical.

## Updating llama.cpp / FastMTP

Do **not** blindly change the pinned llama.cpp commit while continuing to use the same FastMTP patch. Update both together from the model author's current instructions, then rebuild the GHCR image.

The normal embedded-MTP path can be used for troubleshooting without the FastMTP sidecar:

```dotenv
USE_FASTMTP=0
```

## Sources / upstream

- Model and FastMTP instructions: https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Vast CLI docs: https://docs.vast.ai/cli/hello-world
- Vast connection modes: https://docs.vast.ai/guides/instances/connect/overview
- Hugging Face Xet downloads: https://huggingface.co/docs/huggingface_hub/guides/download

Upstream model/patch licensing remains governed by the upstream repository. This deployment repository does not redistribute model weights.
