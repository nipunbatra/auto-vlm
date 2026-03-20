"""Generate samples from LoRA model with visual overlays for OD/Seg."""
import json, torch, timm, re, base64, io
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM
from peft import PeftModel
from prepare_data import load_tokenizer as load_old_tok, load_data, load_images
from train_lora import setup_tokenizer

DEVICE = "cuda"
COLORS = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255),(200,150,50),(50,200,150)]

def img_b64(arr):
    buf = io.BytesIO(); Image.fromarray((arr*255).astype(np.uint8)).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()

def parse_boxes_from_text(text, img_size=224):
    """Parse 'class [BOX: x1 y1 x2 y2]' from prediction text."""
    objects = []
    # Match: word [BOX: num num num num]
    for m in re.finditer(r'(\w+)\s*\[BOX:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', text):
        cls = m.group(1)
        coords = [int(m.group(i))/999.0*img_size for i in range(2,6)]
        objects.append((cls, coords))
    return objects

def parse_polys_from_text(text, img_size=224):
    """Parse 'class [POLY: x1 y1 x2 y2 ...]' from prediction text."""
    objects = []
    for m in re.finditer(r'(\w+)\s*\[POLY:\s*([\d\s]+)\]', text):
        cls = m.group(1)
        nums = [int(x) for x in m.group(2).split()]
        points = [(nums[i]/999.0*img_size, nums[i+1]/999.0*img_size) for i in range(0, len(nums)-1, 2)]
        if len(points) >= 3:
            objects.append((cls, points))
    return objects

def render_boxes(img_arr, objects, title=""):
    img = Image.fromarray((img_arr*255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img)
    for idx, (cls, box) in enumerate(objects[:8]):
        c = COLORS[idx % len(COLORS)]
        draw.rectangle(box, outline=c, width=2)
        draw.text((box[0]+2, max(0,box[1]-12)), cls, fill=c)
    if title: draw.text((2,2), title, fill=(255,255,255))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def render_polys(img_arr, objects, title=""):
    img = Image.fromarray((img_arr*255).astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    for idx, (cls, pts) in enumerate(objects[:8]):
        c = COLORS[idx % len(COLORS)]
        flat = [(int(x),int(y)) for x,y in pts]
        if len(flat) >= 3:
            draw.polygon(flat, outline=c, fill=(*c, 40))
            draw.text((flat[0][0], flat[0][1]-12), cls, fill=c)
    if title:
        draw2 = ImageDraw.Draw(img); draw2.text((2,2), title, fill=(255,255,255))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def fmt(ids, tok):
    tokens = [tok.id_to_token.get(i, "?") for i in ids]
    text = " ".join(tokens)
    for a, b in [("<box>","[BOX: "),("</box>","]"),("<poly>","[POLY: "),("</poly>","]"),("<sep>"," | "),("<eos>",""),("<pad>","")]:
        text = text.replace(a, b)
    return re.sub(r'<loc(\d{3})>', r' \1', re.sub(r'\s+', ' ', text)).strip()

# Load model
old_tok = load_old_tok(); tokenizer = setup_tokenizer()
cfg = json.load(open("checkpoint_lora/config.json"))
vision = timm.create_model(cfg["vision_model"], pretrained=True, num_classes=0).to(DEVICE).eval()
vdim, ldim = cfg["vision_dim"], cfg["lm_dim"]
projector = torch.nn.Sequential(torch.nn.Linear(vdim, ldim), torch.nn.GELU(), torch.nn.Linear(ldim, ldim))
projector.load_state_dict(torch.load("checkpoint_lora/projector.pt", map_location=DEVICE))
projector = projector.to(DEVICE).eval()
lm = AutoModelForCausalLM.from_pretrained(cfg["lm_model"], trust_remote_code=True, dtype=torch.bfloat16)
lm.resize_token_embeddings(len(tokenizer))
lm = PeftModel.from_pretrained(lm, "checkpoint_lora/lora").to(DEVICE).eval()

val_examples = load_data("val"); images = load_images()
sample_ids = json.load(open("data/processed/sample_ids.json"))

samples = {}
for task, picks in sample_ids.items():
    tl = []
    for pick in picks[:3]:
        idx, img_id = pick["index"], pick["image_id"]
        if img_id not in images: continue
        ex = val_examples[idx]; image = images[img_id]
        gt_text = fmt(ex["target_ids"], old_tok)
        
        # Inference
        img_t = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(DEVICE)
        img_t = (img_t - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)
        with torch.no_grad():
            vt = projector(vision.forward_features(img_t)[:,1:,:]).to(torch.bfloat16)
        input_ids = tokenizer.encode(old_tok.decode(ex["input_ids"]), add_special_tokens=False)
        full_emb = torch.cat([vt, lm.get_input_embeddings()(torch.tensor([input_ids], device=DEVICE))], dim=1)
        with torch.no_grad():
            gen = []
            for _ in range(120):
                next_id = lm(inputs_embeds=full_emb).logits[0,-1,:].argmax().item()
                if next_id == tokenizer.eos_token_id: break
                gen.append(next_id)
                full_emb = torch.cat([full_emb, lm.get_input_embeddings()(torch.tensor([[next_id]], device=DEVICE))], dim=1)
        
        pred_raw = tokenizer.decode(gen, skip_special_tokens=False)
        pred_text = pred_raw
        for a, b in [("<box>","[BOX: "),("</box>","]"),("<poly>","[POLY: "),("</poly>","]"),("<sep>"," | ")]:
            pred_text = pred_text.replace(a, b)
        pred_text = re.sub(r'<loc(\d{3})>', r' \1', pred_text)
        
        inp_text = fmt(ex["input_ids"], old_tok) if task=="vqa" else f"<{task}>"
        
        sample = {
            "image_id": img_id, "task": task, "input": inp_text,
            "ground_truth": gt_text, "prediction": pred_text.strip(),
            "image_b64": img_b64(image),
        }
        
        # Render visual overlays for OD and Seg
        if task == "od":
            gt_boxes = parse_boxes_from_text(gt_text)
            pred_boxes = parse_boxes_from_text(pred_text)
            if gt_boxes: sample["gt_viz_b64"] = render_boxes(image, gt_boxes, "GT")
            if pred_boxes: sample["pred_viz_b64"] = render_boxes(image, pred_boxes, "Pred")
        elif task == "seg":
            gt_polys = parse_polys_from_text(gt_text)
            pred_polys = parse_polys_from_text(pred_text)
            if gt_polys: sample["gt_viz_b64"] = render_polys(image, gt_polys, "GT")
            if pred_polys: sample["pred_viz_b64"] = render_polys(image, pred_polys, "Pred")
        
        tl.append(sample)
        print(f"[{task}] img={img_id}: {pred_text[:70]}")
    samples[task] = tl

json.dump({"run": "L07", "samples": samples}, open("samples_torch.json", "w"))
print(f"\nSaved {sum(len(v) for v in samples.values())} samples with viz overlays")
