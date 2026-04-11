---
title: Qwen Image Edit Rapid AIO (NSFW)
emoji: ðŸŒ
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: true
tags:
  - not-for-all-audiences
---

# Qwen Image Edit Rapid AIO on Runpod Serverless

This repo now contains two deployment paths:

- `app.py` keeps the original Hugging Face Space UI flow.
- `handler.py` + `runpod_inference.py` provide a Runpod Serverless worker.

## Runpod files added

- `handler.py`: Runpod entrypoint.
- `runpod_inference.py`: model loading, request parsing, inference, and image serialization.
- `Dockerfile`: container image for Runpod.
- `requirements.runpod.txt`: Python dependencies for the worker image.
- `.dockerignore`: keeps the image build smaller and cleaner.
- `.env.runpod.example`: environment variable template.
- `test_input.json`: local Runpod SDK smoke-test payload.

## Important sizing notes

This model stack is large:

- `Qwen/Qwen-Image-Edit-2511` is listed on Hugging Face at about `57.7 GB`.
- `Phr00t/Qwen-Image-Edit-Rapid-AIO` v23 checkpoint is roughly `28 GB`.

Because of that, use these as your starting assumptions:

- Container disk: at least `120 GB`, with `150 GB` safer.
- GPU: start with an `80 GB` class GPU if you want the highest chance of a first-pass success.
- Workers: start with `max workers = 1` until you confirm memory, boot time, and cost.

## Request format

The Runpod handler accepts payloads like this:

```json
{
  "input": {
    "prompt": "Replace Pikachu's sign text with \"Runpod ready\" while preserving the yarn art style.",
    "images": [
      "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
    ],
    "seed": 42,
    "randomize_seed": false,
    "true_guidance_scale": 1.0,
    "num_inference_steps": 4,
    "rewrite_prompt": false,
    "num_images_per_prompt": 1,
    "height": null,
    "width": null,
    "output_format": "png",
    "upload_to_bucket": false
  }
}
```

### Supported image input keys

- `images`: list of image URLs, local paths, base64 strings, or `{ "url" | "base64" | "path" }` objects.
- `image`, `image_url`, `image_urls`, `image_base64`, `image_base64s`: accepted as aliases.

### Output format

The worker returns:

- `status`
- `seed`
- `prompt`
- `resolved_prompt`
- `images`
- `timings`
- `model`

Each output image contains an `image_url` value. By default this is a data URI (`data:image/...;base64,...`). If you enable bucket uploads, it becomes the uploaded file URL.

## Local smoke test

### 1. Wiring-only test

This checks the Runpod handler flow without downloading the model:

```powershell
$env:RUNPOD_SKIP_MODEL_LOAD="1"
python handler.py --rp_server_api
```

### 2. Real local test

Use this only on a machine with a suitable NVIDIA GPU and enough disk space:

```powershell
pip install -r requirements.runpod.txt
python handler.py --rp_server_api
```

The SDK uses `test_input.json` by default for local testing.

## Local test client

If you want a simple upload-and-prompt UI for testing the deployed Runpod endpoint, use:

- `runpod_test_client.py`
- `requirements.client.txt`

Start it locally:

```powershell
pip install -r requirements.client.txt
python runpod_test_client.py
```

Then open:

```text
http://127.0.0.1:7861
```

The client lets you:

- paste your endpoint ID and Runpod API key
- upload one image
- enter a prompt
- keep `Lock Face Identity` enabled to preserve the source face during edits
- submit the job to Runpod
- preview the returned image
- inspect the raw JSON response

## Step-by-step Runpod deployment

### 1. Put this repo somewhere Runpod can build from

You can use either path:

1. Push this repo to GitHub and use Runpod's GitHub import.
2. Build the Docker image yourself and push it to Docker Hub or another registry.

GitHub import is usually simpler for a first deployment.

### 2. Create a new Serverless endpoint

In Runpod:

1. Open the Serverless section.
2. Choose `New Endpoint`.
3. Import from your Git repository or Docker registry.
4. Keep the endpoint type as `Queue`.

### 3. Point Runpod at the Dockerfile

If you use GitHub import:

- Repository: your fork or copy of this repo.
- Branch: the branch you want to deploy.
- Dockerfile path: `Dockerfile`

### 4. Set the worker hardware

Use conservative settings first:

- GPU: choose a high-VRAM GPU, ideally `80 GB`.
- Min workers: `0`
- Max workers: `1`
- Idle timeout: `300` to `900` seconds is a good starting point for this model.
- Execution timeout: `1800` seconds
- Container disk: `120 GB` minimum, `150 GB` preferred

### 5. Enable Runpod cached models for the base model

In the endpoint settings, set the `Model` field to:

```text
Qwen/Qwen-Image-Edit-2511
```

That tells Runpod to pre-cache the 57.7 GB base model and mount it into the worker at `/runpod-volume/huggingface-cache/hub`.

### 6. Decide how to persist the v23 checkpoint

The base model can use Runpod cached models, but the extra v23 checkpoint is a second Hugging Face artifact.

Recommended setup:

1. Attach a network volume.
2. Set these environment variables so the checkpoint download persists:

```text
HF_HOME=/runpod-volume/hf-home
HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-home/hub
TRANSFORMERS_CACHE=/runpod-volume/hf-home/hub
```

If you skip the network volume, the checkpoint can still download, but future cold starts may have to fetch it again.

### 7. Add environment variables

Start with these:

```text
BASE_MODEL_ID=Qwen/Qwen-Image-Edit-2511
CHECKPOINT_REPO_ID=Phr00t/Qwen-Image-Edit-Rapid-AIO
CHECKPOINT_FILENAME=v23/Qwen-Rapid-AIO-NSFW-v23.safetensors
DEFAULT_NUM_INFERENCE_STEPS=4
DEFAULT_TRUE_GUIDANCE_SCALE=1.0
DEFAULT_REWRITE_PROMPT=false
RUNPOD_USE_CACHED_BASE_MODEL=true
RUNPOD_ENABLE_BUCKET_UPLOADS=false
RUNPOD_INIT_TIMEOUT=1800
```

Optional:

- `HF_TOKEN`: only needed for gated/private Hugging Face assets.
- `HF_INFERENCE_API_KEY`: only needed if you want `rewrite_prompt=true`.
- `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`: only needed if you want uploaded output URLs instead of base64 data URIs.

You can copy these from `.env.runpod.example`.

### 8. Deploy and watch the first boot

Click deploy, then watch:

- Build logs: confirms the image built successfully.
- Worker logs: confirms base model load, checkpoint injection, and first job execution.

The first worker boot will be slow because it has to:

1. Start the container.
2. Load the cached base model.
3. Download the v23 checkpoint if it is not already present.
4. Inject the checkpoint weights into the pipeline.

### 9. Send a synchronous test request

Replace `YOUR_ENDPOINT_ID` and `YOUR_API_KEY`:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

If you prefer async:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

Then poll:

```bash
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 10. Decode or consume the output

By default, each result image is returned as a data URI in:

```text
output.images[0].image_url
```

If you enable bucket uploads, the same field contains a normal URL instead.

## Deployment advice that matters for this repo

- Leave `rewrite_prompt` disabled unless you really want it. It adds an external dependency and extra latency.
- Keep `max workers = 1` until you know the model fits comfortably on your chosen GPU type.
- Prefer a longer idle timeout for this worker than you would for a small model. Cold starts are expensive here.
- If boot time is still too high, keep one active worker warm instead of relying only on flex workers.

## Files you will usually touch

- `Dockerfile`: change the base image or baked defaults.
- `requirements.runpod.txt`: adjust Python packages.
- `runpod_inference.py`: change input schema, output schema, model loading, or checkpoint logic.
- `.env.runpod.example`: keep your environment variable reference up to date.

## Existing Hugging Face Space support

The original Gradio application remains in `app.py`. The Runpod files are additive and do not remove the original Space path.
