import streamlit as st
import os
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

import torch
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from pathlib import Path

# load image from the IAM database (actually this model is meant to be used on printed text)

st.title("Local LLM with Image to Text processing")
st.write("This page allows you to select an image and analyze it using a local LLM model.")
st.write("With faster GPUs and more memory, these processes run rather quickly and can make short work of things like invoices, receipts, etc.")
st.write("This takes a LONG time (~10 mins after clicking the button) and the output depends on the resolution of the image, so please be patient.")

image_walk = list(os.walk("./data/images"))
image_names = image_walk[0][2]
root_path = image_walk[0][0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Selected device:", DEVICE)

selected_image = st.selectbox("Select an image", image_names, key="image_select")

if selected_image:
    st.image(f"{root_path}/{selected_image}", caption=selected_image)


if st.button("Analyze Text"):

    with st.spinner("Analyzing text, this may take ~10 minutes!"):

        print("Initializing processor and model...")
        processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        ).to(DEVICE)

        print("Creating input messages...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."},
                ],
            },
        ]

        print("Preparing inputs...")
        image = load_image(f"{root_path}/{selected_image}")
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)

        print("Generating outputs...")
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        doctags = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()

        print("Populating document...")
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        print(doctags)

        print("Creating DoclingDocument...")
        doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")

        st.write(doc.export_to_markdown())


# for root, folders, files in all_images:
#     for file in files:
#         st.write(f"{root}/{file}")
#         st.image(f"{root}/{file}")
#         image_path = f"{root}/{file}"

#         result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
#         print(result)
