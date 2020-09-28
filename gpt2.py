from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("gpt2-xl")
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

while(True):
    file_name = input("Enter a file with a prompt: ")
    with open(file_name, "r") as f:
        prompt = f.read()

    for i in range(3):
        print("Generating output", str(i), "for prompt:", prompt)
        print("-------------------------------------------------------")
        print("Output: ")
        inputs = tokenizer.encode( prompt, add_special_tokens=False, return_tensors="pt")

        prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        outputs = model.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=60)
        output = prompt + tokenizer.decode(outputs[0])[prompt_length:]

        print(output)

        with open(file_name, "a") as f:
            f.write("\n------------------------------------------------------\n")
            f.write("Output:\n")
            f.write(output)
            f.write("\n")



def adjust_prompt(prompt):
    