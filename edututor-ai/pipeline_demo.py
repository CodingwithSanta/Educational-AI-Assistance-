from transformers import pipeline

def main():
    pipe = pipeline("text-generation", model="ibm-granite/granite-3.2-2b-instruct")
    messages = [{"role": "user", "content": "Who are you?"}]
    result = pipe(messages, max_new_tokens=40)
    print(result)

if __name__ == "__main__":
    main()
