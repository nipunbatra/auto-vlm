#!/usr/bin/env python3
"""
Gradio interactive demo for Auto-VLM.
Upload an image, pick a task, get predictions.

Run: python demo.py
Or: gradio demo.py
"""

import json
import re
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Global model references (loaded once)
_model = None
_tokenizer = None
_vision = None
_projector = None
_lm = None
_old_tok = None


def load_model():
    """Load the LoRA model (once)."""
    global _tokenizer, _vision, _projector, _lm, _old_tok

    if _lm is not None:
        return

    import timm
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from prepare_data import load_tokenizer as load_old_tok

    _old_tok = load_old_tok()

    # Determine model from checkpoint config
    config_path = Path("checkpoint_lora/config.json")
    if config_path.exists():
        cfg = json.load(open(config_path))
    else:
        cfg = {"vision_model": "vit_small_patch16_224", "lm_model": "Qwen/Qwen2.5-0.5B",
               "vision_dim": 384, "lm_dim": 896}

    from train_lora import setup_tokenizer
    _tokenizer = setup_tokenizer()

    print(f"Loading vision: {cfg['vision_model']}...")
    _vision = timm.create_model(cfg["vision_model"], pretrained=True, num_classes=0).to(DEVICE).eval()

    print(f"Loading projector...")
    vdim = cfg.get("vision_dim", 384)
    ldim = cfg.get("lm_dim", 896)
    _projector = torch.nn.Sequential(
        torch.nn.Linear(vdim, ldim),
        torch.nn.GELU(),
        torch.nn.Linear(ldim, ldim),
    )
    _projector.load_state_dict(torch.load("checkpoint_lora/projector.pt", map_location=DEVICE))
    _projector = _projector.to(DEVICE).eval()

    print(f"Loading LM: {cfg['lm_model']} + LoRA...")
    _lm = AutoModelForCausalLM.from_pretrained(
        cfg["lm_model"], trust_remote_code=True, dtype=torch.bfloat16
    )
    _lm.resize_token_embeddings(len(_tokenizer))
    _lm = PeftModel.from_pretrained(_lm, "checkpoint_lora/lora").to(DEVICE).eval()
    print("Model loaded!")


def predict(image, task, question=""):
    """Run inference on an image with a given task."""
    load_model()

    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB").resize((224, 224))
    img_arr = np.array(image, dtype=np.float32) / 255.0

    # Prepare image tensor
    img_t = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
    img_t = (img_t - mean) / std

    # Vision features
    with torch.no_grad():
        vf = _vision.forward_features(img_t)[:, 1:, :]
        vt = _projector(vf).to(torch.bfloat16)

    # Build input text
    task_map = {
        "Caption": "<caption>",
        "VQA": "<vqa>",
        "Object Detection": "<od>",
        "Segmentation": "<seg>",
        "Describe": "<describe>",
    }
    input_text = task_map.get(task, "<caption>")
    if task == "VQA" and question:
        input_text += f" {question} <sep>"

    input_ids = _tokenizer.encode(input_text, add_special_tokens=False)
    inp_emb = _lm.get_input_embeddings()(torch.tensor([input_ids], device=DEVICE))
    full_emb = torch.cat([vt, inp_emb], dim=1)

    # Generate
    with torch.no_grad():
        gen = []
        for _ in range(150):
            out = _lm(inputs_embeds=full_emb)
            next_id = out.logits[0, -1, :].argmax().item()
            if next_id == _tokenizer.eos_token_id:
                break
            gen.append(next_id)
            next_emb = _lm.get_input_embeddings()(torch.tensor([[next_id]], device=DEVICE))
            full_emb = torch.cat([full_emb, next_emb], dim=1)

    # Decode
    pred = _tokenizer.decode(gen, skip_special_tokens=False)
    # Clean up
    if "<eos>" in pred:
        pred = pred[:pred.index("<eos>")]
    pred = re.sub(r'<unk>|<pad>|<bos>', '', pred)
    pred = pred.replace("<box>", "[BOX: ").replace("</box>", "]")
    pred = pred.replace("<poly>", "[POLY: ").replace("</poly>", "]")
    pred = pred.replace("<sep>", " | ")
    pred = re.sub(r'<loc(\d{3})>', r' \1', pred)
    pred = re.sub(r'\s+', ' ', pred).strip()

    # For OD/Seg, also draw on image
    annotated = None
    if task == "Object Detection":
        annotated = draw_boxes(img_arr, pred)
    elif task == "Segmentation":
        annotated = draw_polys(img_arr, pred)

    return pred, annotated


def draw_boxes(img_arr, text):
    """Draw predicted boxes on image."""
    from PIL import Image, ImageDraw
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    colors = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255)]
    for idx, m in enumerate(re.finditer(r'(\w+)\s*\[BOX:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', text)):
        cls = m.group(1)
        coords = [int(m.group(i)) / 999.0 * 224 for i in range(2, 6)]
        c = colors[idx % len(colors)]
        draw.rectangle(coords, outline=c, width=2)
        draw.text((coords[0]+2, max(0, coords[1]-12)), cls, fill=c)
    return np.array(img)


def draw_polys(img_arr, text):
    """Draw predicted polygons on image."""
    from PIL import Image, ImageDraw
    img = Image.fromarray((img_arr * 255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    colors = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255)]
    for idx, m in enumerate(re.finditer(r'(\w+)\s*\[POLY:\s*([\d\s]+)\]', text)):
        cls = m.group(1)
        nums = [int(x) for x in m.group(2).split()]
        pts = [(nums[i]/999*224, nums[i+1]/999*224) for i in range(0, len(nums)-1, 2)]
        c = colors[idx % len(colors)]
        if len(pts) >= 3:
            draw.polygon([(int(x), int(y)) for x, y in pts], outline=c, fill=(*c, 40))
            draw.text((int(pts[0][0]), int(pts[0][1])-12), cls, fill=c)
    return np.array(img)


def create_demo():
    """Create Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Auto-VLM Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Auto-VLM: Multi-Task Vision-Language Model")
        gr.Markdown("Upload an image and select a task. Model: SigLIP + Qwen2.5 (LoRA)")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image")
                task_dropdown = gr.Dropdown(
                    choices=["Caption", "VQA", "Object Detection", "Segmentation", "Describe"],
                    value="Caption", label="Task"
                )
                question_input = gr.Textbox(
                    label="Question (VQA only)", placeholder="How many dogs are in the image?",
                    visible=True
                )
                submit_btn = gr.Button("Run", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(label="Model Output", lines=5)
                image_output = gr.Image(label="Annotated Image (OD/Seg)")

        submit_btn.click(
            fn=predict,
            inputs=[image_input, task_dropdown, question_input],
            outputs=[text_output, image_output],
        )

        gr.Examples(
            examples=[
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg", "Caption", ""],
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg", "Object Detection", ""],
                ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg", "VQA", "What animal is this?"],
            ],
            inputs=[image_input, task_dropdown, question_input],
        )

    return demo


if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        # CLI mode for testing
        from PIL import Image
        load_model()
        img = Image.open(sys.argv[1] if len(sys.argv) > 2 else "data/images/val2017/000000000139.jpg")
        for task in ["Caption", "Object Detection", "VQA", "Segmentation", "Describe"]:
            q = "What objects are in the image?" if task == "VQA" else ""
            pred, _ = predict(img, task, q)
            print(f"\n{task}: {pred}")
    else:
        demo = create_demo()
        demo.launch(share=True)
