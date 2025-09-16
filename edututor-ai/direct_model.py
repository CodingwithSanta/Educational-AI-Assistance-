from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "ibm-granite/granite-3.2-2b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    messages = [{"role": "user", "content": "Who are you?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40)

    reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    print(reply)\

if __name__ == "__main__":
    main()
