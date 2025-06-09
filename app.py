from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image

def convert_pdf_to_images(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi)
    return images  # List of PIL images
  
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tải mô hình và processor từ Hugging Face
processor = AutoProcessor.from_pretrained("5CD-AI/Vintern-1B-v3_5")
model = AutoModelForVision2Seq.from_pretrained("5CD-AI/Vintern-1B-v3_5").to(device)

def ocr_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def ocr_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    results = []
    for idx, image in enumerate(images):
        print(f"Processing page {idx+1}/{len(images)}...")
        text = ocr_image(image)
        results.append(text)
    return "\n\n".join(results)

pdf_path = "your_file.pdf"
result_text = ocr_pdf(pdf_path)

with open("ocr_output.txt", "w", encoding="utf-8") as f:
    f.write(result_text)
