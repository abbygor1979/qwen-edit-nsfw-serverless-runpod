from __future__ import annotations

import base64
import json
import os
import time
from io import BytesIO
from typing import Any, Dict, Tuple

import gradio as gr
import requests
from PIL import Image


DEFAULT_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "biqd9c2lr7dqjn")
DEFAULT_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DEFAULT_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2")
DEFAULT_POLL_INTERVAL = float(os.environ.get("RUNPOD_POLL_INTERVAL", "5"))
DEFAULT_TIMEOUT = int(os.environ.get("RUNPOD_JOB_TIMEOUT", "900"))


def image_to_data_uri(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{encoded}"


def data_uri_to_image(image_url: str) -> Image.Image:
    payload = image_url.split(",", 1)[1] if image_url.startswith("data:image/") else image_url
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def remote_url_to_image(image_url: str) -> Image.Image:
    response = requests.get(image_url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def build_payload(
    image: Image.Image,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    true_guidance_scale: float,
    rewrite_prompt: bool,
    lock_face_identity: bool,
    num_images_per_prompt: int,
    output_format: str,
) -> Dict[str, Any]:
    return {
        "input": {
            "prompt": prompt,
            "images": [{"base64": image_to_data_uri(image.convert("RGB"), image_format="PNG")}],
            "seed": seed,
            "randomize_seed": False,
            "true_guidance_scale": true_guidance_scale,
            "num_inference_steps": num_inference_steps,
            "rewrite_prompt": rewrite_prompt,
            "lock_face_identity": lock_face_identity,
            "num_images_per_prompt": num_images_per_prompt,
            "output_format": output_format,
        }
    }


def submit_job(
    endpoint_id: str,
    api_key: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{DEFAULT_API_BASE}/{endpoint_id}/run",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def poll_job(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        response = requests.get(
            f"{DEFAULT_API_BASE}/{endpoint_id}/status/{job_id}",
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return data

        time.sleep(poll_interval)

    raise TimeoutError(f"Job {job_id} did not finish within {timeout_seconds} seconds.")


def extract_image(result: Dict[str, Any]) -> Image.Image:
    output = result.get("output", {})
    images = output.get("images", [])
    if not images:
        raise ValueError("No images found in the Runpod response.")

    image_url = images[0].get("image_url")
    if not image_url:
        raise ValueError("The first output image did not contain an image_url.")

    if image_url.startswith("data:image/"):
        return data_uri_to_image(image_url)

    return remote_url_to_image(image_url)


def run_inference(
    endpoint_id: str,
    api_key: str,
    image: Image.Image,
    prompt: str,
    seed: int,
    num_inference_steps: int,
    true_guidance_scale: float,
    rewrite_prompt: bool,
    lock_face_identity: bool,
    num_images_per_prompt: int,
    output_format: str,
) -> Tuple[Image.Image | None, str, str]:
    if not endpoint_id.strip():
        raise gr.Error("Endpoint ID is required.")
    if not api_key.strip():
        raise gr.Error("Runpod API key is required.")
    if image is None:
        raise gr.Error("Upload an image first.")
    if not prompt.strip():
        raise gr.Error("Prompt is required.")

    payload = build_payload(
        image=image,
        prompt=prompt,
        seed=int(seed),
        num_inference_steps=int(num_inference_steps),
        true_guidance_scale=float(true_guidance_scale),
        rewrite_prompt=bool(rewrite_prompt),
        lock_face_identity=bool(lock_face_identity),
        num_images_per_prompt=int(num_images_per_prompt),
        output_format=output_format.lower(),
    )

    try:
        submitted = submit_job(endpoint_id.strip(), api_key.strip(), payload)
        job_id = submitted["id"]
        result = poll_job(endpoint_id.strip(), api_key.strip(), job_id)
        output_image = extract_image(result) if result.get("status") == "COMPLETED" else None
        status_text = (
            f"Job {job_id}\n"
            f"Status: {result.get('status')}\n"
            f"Delay: {result.get('delayTime', 'n/a')} ms\n"
            f"Execution: {result.get('executionTime', 'n/a')} ms"
        )
        return output_image, status_text, json.dumps(result, indent=2)
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        raise gr.Error(f"Runpod request failed: {body}") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(title="Runpod Qwen Image Test Client") as demo:
    gr.Markdown("# Runpod Qwen Image Test Client")
    gr.Markdown("Upload one image, enter a prompt, and test your Runpod endpoint.")

    with gr.Row():
        endpoint_id = gr.Textbox(label="Endpoint ID", value=DEFAULT_ENDPOINT_ID)
        api_key = gr.Textbox(label="Runpod API Key", value=DEFAULT_API_KEY, type="password")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Output Image", type="pil")

    prompt = gr.Textbox(
        label="Prompt",
        lines=3,
        placeholder='Example: Replace the sign text with "Runpod ready" while preserving the style.',
    )

    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            seed = gr.Number(label="Seed", value=42, precision=0)
            num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=10, step=1, value=4)
            true_guidance_scale = gr.Slider(label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)

        with gr.Row():
            rewrite_prompt = gr.Checkbox(label="Rewrite Prompt", value=False)
            lock_face_identity = gr.Checkbox(label="Lock Face Identity", value=True)
            num_images_per_prompt = gr.Slider(label="Images Per Prompt", minimum=1, maximum=4, step=1, value=1)
            output_format = gr.Dropdown(label="Output Format", choices=["png", "jpeg"], value="png")

    run_button = gr.Button("Run Test", variant="primary")
    status_box = gr.Textbox(label="Status", lines=4)
    raw_response = gr.Code(label="Raw Runpod Response", language="json")

    run_button.click(
        fn=run_inference,
        inputs=[
            endpoint_id,
            api_key,
            input_image,
            prompt,
            seed,
            num_inference_steps,
            true_guidance_scale,
            rewrite_prompt,
            lock_face_identity,
            num_images_per_prompt,
            output_format,
        ],
        outputs=[output_image, status_box, raw_response],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("RUNPOD_TEST_CLIENT_PORT", "7861")))
