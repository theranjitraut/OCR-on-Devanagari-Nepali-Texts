'''
pymupdf
torch
transformers
accelerate
bitsandbytes
Pillow
tqdm
'''

import os
'''
Just in case of GPU issue
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
'''

import gc
from pathlib import Path

import torch
# import pymupdf
import fitz  # PyMuPDF; also if it doesn't work then use import mpymupdf
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Configuration
INPUT_DIR = Path("./pdf_input")
OUTPUT_DIR = Path("./text_output")
# ./models/qwen3.5-4b
# "Qwen/Qwen3.5-4B"
MODEL_ID = Path("./models/qwen3.5-4b")    # HF repo id, or a local folder path
DPI = 150 # it seems 200 is good dpi, but starting from 150
MAX_DIMENSION = 1280               # longer side of each page image, in px
MAX_NEW_TOKENS = 2048

OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL Nepali (Devanagari script) text "
    "visible in this image exactly as written, preserving line breaks and "
    "reading order. Do not translate, summarize, or add commentary. "
    "Do not include page numbers, running headers, or footers - transcribe "
    "only the main body text. Output only the transcribed text, nothing else."
)


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
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
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_PROMPT},
        ],
    }]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat_text], images=[image], return_tensors="pt").to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    trimmed = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("Warning: no CUDA GPU detected — 4-bit inference needs one.")

    print(f"Loading {MODEL_ID} in 4-bit ...")
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
                torch.cuda.empty_cache() # to empty cache

        # No page numbers/markers in the saved text — just the transcribed body content, one page's text after another.
        out_path.write_text("\n\n".join(page_texts), encoding="utf-8")
        print(f"  saved -> {out_path}")

if __name__ == "__main__":
    main()
