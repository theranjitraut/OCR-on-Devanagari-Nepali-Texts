# Importing necessary dependencies
import os
from pathlib import Path

from mistralai import Mistral
# pip install mistralai

# Configuration
INPUT_DIR = Path("./pdf_input")
OUTPUT_DIR = Path("./ocr_output")
MISTRAL_MODEL = "mistral-ocr-latest"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

def ocr_pdf(client: Mistral, pdf_path: Path) -> str:
    uploaded = client.files.upload(
        file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)
    response = client.ocr.process(
        model=MISTRAL_MODEL,
        document={"type": "document_url", "document_url": signed_url.url},
    )
    return "\n\n".join(page.markdown for page in response.pages)


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MISTRAL_API_KEY:
        raise RuntimeError("Set the MISTRAL_API_KEY environment variable first.")

    client = client.Mistral(api_key=MISTRAL_API_KEY)

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
        text = ocr_pdf(client, pdf_path)
        out_path.write_text(text, encoding="utf-8")
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
