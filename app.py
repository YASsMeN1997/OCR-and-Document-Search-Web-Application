import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import requests
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from io import BytesIO
import torch
import re
import base64

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", verbose=10)
model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

def create_rag_index(image_path):
    RAG.index(
        input_path=image_path,
        index_name="image_index",
        store_collection_with_index=True,
        overwrite=True,
    )

def extract_relevant_text(qwen_output):
    # Extract the main content from the Qwen2-VL output (assuming it's a list of strings)
    qwen_text = qwen_output[0]

    # Split the text by newlines and periods to handle various sentence structures
    lines = qwen_text.split('\n')

    # Initialize a list to hold relevant text lines
    relevant_text = []

    # Loop through each line to identify relevant text
    for line in lines:
        # Use a regex to match text that looks like it's extracted from the image
        # We ignore any description or meta information
        if re.match(r'[A-Za-z0-9]', line):  # Matches lines that have words or numbers
            relevant_text.append(line.strip())

    # Join the relevant text into a single output (you can customize the format)
    return "\n".join(relevant_text)


# put all in one function
def ocr_image(image_path,text_query):
    if text_query:
      create_rag_index(image_path)
      results = RAG.search(text_query, k=1, return_base64_results=True)

      image_data = base64.b64decode(results[0].base64)
      image = Image.open(BytesIO(image_data))
    else:
      image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": "explain all text find in the image."
                }
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    inputs = inputs.to(device)

    output_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Extract relevant text from the Qwen2-VL output
    relevant_text = extract_relevant_text(output_text)

    return relevant_text


def highlight_text(text, query):
    highlighted_text = text
    for word in query.split():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_text = pattern.sub(lambda m: f'<span style="background-color: yellow;">{m.group()}</span>', highlighted_text)
    return highlighted_text

def ocr_and_search(image, keyword):
    extracted_text = ocr_image(image,keyword)
    #print(extracted_text)
    if keyword =='':
      return extracted_text , 'Please Enter a Keyword'

    else:
      highlighted_text = highlight_text(extracted_text, keyword)
    return extracted_text , highlighted_text

# Create Gradio Interface
interface = gr.Interface(
    fn=ocr_and_search,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Textbox(label="Enter Keyword")
    ],
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.HTML("Search Result"),
    ],
    title="OCR and Document Search Web Application",
    description="Upload an image to extract text in Hindi and English and search for keywords."
)

if __name__ == "__main__":
    interface.launch(share=True)
