import gradio as gr
import os
import subprocess
import shutil
import json
import time
from pathlib import Path
import torch
import spaces
from diffusers import DiffusionPipeline

# ==========================================
# 1. SETUP & GLOBAL VARS
# ==========================================

DATASET_DIR = Path("./datasets")
OUTPUT_DIR = Path("./output")
DATASET_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# global tracking for loras
# key: friendly name, value: path
AVAILABLE_LORAS = {} 

print("loading z-image-turbo pipeline...")
pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")
print("pipeline loaded!")

# ==========================================
# 2. TRAINING LOGIC
# ==========================================

def check_gpu():
    if torch.cuda.is_available():
        return f"✅ gpu available: {torch.cuda.get_device_name(0)}"
    return "⚠️ no gpu detected"

def upload_and_prepare_dataset(files, dataset_name, trigger_word):
    if not files:
        return "❌ upload images first", None, ""
    
    if not dataset_name:
        dataset_name = f"dataset_{int(time.time())}"
    
    dataset_path = DATASET_DIR / dataset_name
    dataset_path.mkdir(exist_ok=True, parents=True)
    
    image_count = 0
    for file in files:
        if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            filename = Path(file.name).name
            dest = dataset_path / filename
            shutil.copy(file.name, dest)
            
            caption_file = dest.with_suffix('.txt')
            caption_text = trigger_word if trigger_word else "a photo"
            with open(caption_file, 'w') as f:
                f.write(caption_text)
            
            image_count += 1
            
    if image_count == 0:
        return "❌ no valid images found", None, ""
        
    return f"✅ ready: {image_count} images in {dataset_name}", str(dataset_path), dataset_name

# request 10 mins gpu for training
@spaces.GPU(duration=600) 
def train_lora(
    dataset_path,
    project_name,
    trigger_word,
    steps,
    learning_rate,
    lora_rank,
    resolution,
    progress=gr.Progress()
):
    if not dataset_path:
        return "❌ no dataset", None

    if not project_name:
        project_name = f"lora_{int(time.time())}"
        
    output_path = OUTPUT_DIR / project_name
    output_path.mkdir(exist_ok=True, parents=True)
    
    # config generation
    config = {
        "job": "extension",
        "config": {
            "name": project_name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": str(output_path),
                "device": "cuda:0",
                "trigger_word": trigger_word or "",
                "network": {
                    "type": "lora",
                    "linear": int(lora_rank),
                    "linear_alpha": int(lora_rank),
                },
                "save": {
                    "dtype": "float16",
                    "save_every": int(steps), # save only at end to save space
                    "max_step_saves_to_keep": 1,
                },
                "datasets": [{
                    "folder_path": dataset_path,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "resolution": [int(resolution), int(resolution)],
                }],
                "train": {
                    "batch_size": 1,
                    "steps": int(steps),
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "lr": float(learning_rate),
                    "ema_config": {"use_ema": True, "ema_decay": 0.99},
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": "Tongyi-MAI/Z-Image-Base",
                    "is_v_pred": False,
                    "quantize": True,
                },
            }]
        }
    }
    
    config_path = output_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    # install ai-toolkit
    progress(0.1, desc="setting up environment...")
    if not Path("./ai-toolkit").exists():
        try:
            subprocess.run(["git", "clone", "https://github.com/ostris/ai-toolkit.git"], check=True)
            subprocess.run(["pip", "install", "-q", "-r", "ai-toolkit/requirements.txt"], check=True)
        except Exception as e:
            return f"❌ setup failed: {e}", None

    progress(0.2, desc="training (this takes time)...")
    
    try:
        # run training script
        # explicitly passing environment to ensure cuda visibility in subprocess
        env = os.environ.copy()
        
        proc = subprocess.run(
            ["python", "ai-toolkit/run.py", str(config_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=3500 
        )
        
        if proc.returncode != 0:
            return f"❌ training crashed:\n{proc.stderr}", None
            
        # find result
        lora_files = list(output_path.glob("*.safetensors"))
        if lora_files:
            lora_file = lora_files[-1]
            AVAILABLE_LORAS[project_name] = str(lora_file)
            
            # update the dropdown choices dynamically
            choices = [("None", None)] + [(k, v) for k, v in AVAILABLE_LORAS.items()]
            
            return f"✅ trained: {project_name}", str(lora_file)
        
        return "⚠️ finished but no safetensors found", None

    except Exception as e:
        return f"❌ fatal error: {e}", None

# ==========================================
# 3. INFERENCE LOGIC
# ==========================================

@spaces.GPU
def generate_image(
    prompt, 
    height, 
    width, 
    steps, 
    seed, 
    randomize_seed, 
    lora_path,
    lora_scale
):
    # handle lora loading/unloading
    pipe.unload_lora_weights() # clean slate
    
    if lora_path and os.path.exists(lora_path):
        print(f"loading lora: {lora_path}")
        try:
            pipe.load_lora_weights(lora_path)
            # manual scaling not always supported directly without fuse, 
            # but usually applied by default. 
            # for simplicitly we just load it.
        except Exception as e:
            print(f"lora load failed: {e}")

    if randomize_seed:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generator = torch.Generator("cuda").manual_seed(int(seed))
    
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    return image, seed

def update_lora_list():
    """helper to refresh dropdown"""
    choices = [("None", None)] + [(k, v) for k, v in AVAILABLE_LORAS.items()]
    return gr.Dropdown(choices=choices)

# ==========================================
# 4. UI CONSTRUCTION
# ==========================================

custom_theme = gr.themes.Soft(primary_hue="yellow", secondary_hue="slate")

with gr.Blocks(theme=custom_theme, title="Z-Image ZeroGPU Trainer") as demo:
    
    gr.Markdown("# ⚡ Z-Image-Turbo: Train & Test")
    
    with gr.Tabs():
        
        # TAB 1: INFERENCE
        with gr.Tab("🎨 Generate"):
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(label="Prompt", lines=3)
                    
                    with gr.Row():
                        lora_selector = gr.Dropdown(
                            label="Select LoRA",
                            choices=[("None", None)],
                            value=None,
                            interactive=True
                        )
                        refresh_btn = gr.Button("🔄", size="sm", scale=0)
                    
                    with gr.Accordion("Settings", open=False):
                        h_slider = gr.Slider(512, 2048, 1024, step=64, label="Height")
                        w_slider = gr.Slider(512, 2048, 1024, step=64, label="Width")
                        steps_slider = gr.Slider(1, 50, 9, step=1, label="Steps")
                        seed_num = gr.Number(42, label="Seed")
                        rand_seed = gr.Checkbox(True, label="Randomize Seed")

                    gen_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    out_img = gr.Image(label="Result")
                    out_seed = gr.Number(label="Seed Used")

        # TAB 2: TRAINING
        with gr.Tab("🏋️ Train LoRA"):
            gr.Markdown("⚠️ **Note:** Requires paid GPU space for long timeouts.")
            
            with gr.Row():
                with gr.Column():
                    train_files = gr.Files(label="Images", file_types=["image"])
                    train_name = gr.Textbox(label="Project Name", value="my_lora")
                    train_trigger = gr.Textbox(label="Trigger Word", value="ohwx")
                    
                    # hidden state for dataset path
                    dataset_path_state = gr.State()
                    
                    upload_btn = gr.Button("1. Process Dataset")
                    upload_status = gr.Textbox(label="Dataset Status")
                    
                    gr.Markdown("---")
                    
                    train_steps = gr.Slider(100, 2000, 500, step=100, label="Steps")
                    train_lr = gr.Slider(1e-5, 1e-3, 1e-4, step=1e-5, label="Learning Rate")
                    train_rank = gr.Slider(4, 128, 16, step=4, label="Rank")
                    
                    start_train_btn = gr.Button("2. Start Training", variant="stop")
                
                with gr.Column():
                    train_log = gr.Textbox(label="Training Log", lines=10)
                    lora_file_download = gr.File(label="Download LoRA")

    # WIRING
    
    # Refresh LoRA list
    refresh_btn.click(update_lora_list, outputs=lora_selector)
    
    # Upload
    upload_btn.click(
        upload_and_prepare_dataset,
        [train_files, train_name, train_trigger],
        [upload_status, dataset_path_state, train_name]
    )
    
    # Train
    def on_train_complete(status, file_path):
        # Update available loras list immediately after training
        new_choices = [("None", None)] + [(k, v) for k, v in AVAILABLE_LORAS.items()]
        return status, file_path, gr.Dropdown(choices=new_choices)

    start_train_btn.click(
        train_lora,
        [dataset_path_state, train_name, train_trigger, train_steps, train_lr, train_rank, h_slider], # reusing h_slider for res
        [train_log, lora_file_download]
    ).then(
        update_lora_list,
        outputs=[lora_selector]
    )
    
    # Generate
    gen_btn.click(
        generate_image,
        [prompt_input, h_slider, w_slider, steps_slider, seed_num, rand_seed, lora_selector, train_lr], # train_lr dummy
        [out_img, out_seed]
    )

if __name__ == "__main__":
    demo.launch()