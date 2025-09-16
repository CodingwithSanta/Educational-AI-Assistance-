import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.replace(prompt, "").strip()

def concept_explanation(concept):
    return generate_response(f"Explain the concept of {concept} in detail with examples:", max_length=800)

def quiz_generator(concept):
    return generate_response(
        f"Generate 5 quiz questions about {concept} with different question types. "
        "At the end, provide all the answers in an ANSWERS section:",
        max_length=1000
    )

with gr.Blocks() as app:
    gr.Markdown("# 📘 Educational AI Assistant")
    with gr.Tabs():
        with gr.TabItem("Concept Explanation"):
            concept = gr.Textbox(label="Enter a concept")
            out1 = gr.Textbox(label="Explanation", lines=10)
            gr.Button("Explain").click(concept_explanation, inputs=concept, outputs=out1)

        with gr.TabItem("Quiz Generator"):
            topic = gr.Textbox(label="Enter a topic")
            out2 = gr.Textbox(label="Quiz", lines=15)
            gr.Button("Generate Quiz").click(quiz_generator, inputs=topic, outputs=out2)

if __name__ == "__main__":
    app.launch(share=True)
