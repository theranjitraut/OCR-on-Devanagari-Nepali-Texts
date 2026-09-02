# Importing necessary libraries
import os
import gc
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Configuration
INPUT_DIR = Path("./pdf_input")
OUTPUT_DIR = Path("./ocr_output")
MODEL_ID = "PaddlePaddle/PaddleOCR-VL"   # HF repo id, or a local folder path
DPI = 200
MAX_DIMENSION = 1280                     # longer side of each page image, in px
MAX_NEW_TOKENS = 2048
PROMPT = "OCR:"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor

def pdf_to_images(pdf_path: Path) -> list[Image.Image]:
    zoom = DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            w, h = img.size
            longest = max(w, h)
            if longest > MAX_DIMENSION:
                scale = MAX_DIMENSION / longest
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            images.append(img)
    return images

@torch.inference_mode()
def ocr_image(model, processor, image: Image.Image) -> str:
    messages = [{"role": "user", "content": PROMPT}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inputs.items()}

    generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — this model needs one.")

    print(f"Loading {MODEL_ID} ...")
    model, processor = load_model()
    print("Model loaded.")

    pdf_files = sorted(INPUT_DIR.glob("*.pdf")) + sorted(INPUT_DIR.glob("*.PDF"))
    if not pdf_files:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    for pdf_path in pdf_files:
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        if out_path.exists():
            print(f"Skipping (already done): {pdf_path.name}")
            continue

        print(f"Processing: {pdf_path.name}")
        pages = pdf_to_images(pdf_path)
        page_texts = []
        for i, page_img in enumerate(pages, start=1):
            print(f"  page {i}/{len(pages)}")
            page_texts.append(ocr_image(model, processor, page_img))
            del page_img
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_path.write_text("\n\n".join(page_texts), encoding="utf-8")
        print(f"saved -> {out_path}")

if __name__ == "__main__":
    main()
