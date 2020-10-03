
from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("gpt2-xl")#.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

"""
This is the typical way of generating output
note: the prompt is included in n_tokens
"""
def generate_n_tokens(prompt, n_tokens):
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    outputs = model.generate(inputs, max_length=n_tokens, do_sample=True, top_p=0.95, top_k=60)
    return prompt + tokenizer.decode(outputs[0])[prompt_length:]

"""
very computationally wastful
"""
def generate_next_word(prompt, max_tokens):
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    outputs = model.generate(inputs, max_length=max_tokens, do_sample=True, top_p=0.95, top_k=60)
    output = prompt + tokenizer.decode(outputs[0])[prompt_length:]

    new_output = output[len(prompt):]
    new_words = new_output.split()
    first_new_word = new_words[0]
    return first_new_word



def get_current_token_length(text):
    tokenized_text =  tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    return len(tokenizer.decode(tokenized_text[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))

"""
n is the maximum number of words that the original prompt
can ever be from the end of the current prompt. words will be deleted
from the middle if this condition is violated. This is done in an attempt
to allow the original prompt to exhert a greater influence on the output.
"""
def generate_with_prompt(prompt):
    output = prompt
    current_prompt = prompt
    output_length = 0
    while(output_length < 450):
        next_word = generate_next_word(current_prompt, get_current_token_length(current_prompt) + 5)

        # Append the new word to the end of the current prompt and the output
        output += " " + next_word
        print(next_word)

        # If the original prompt is too far from the end of the current prompt adjust it
        current_prompt = adjust_prompt(current_prompt, prompt, 15)
        print("current prompt: ")
        print(current_prompt)
        print("----------------------------")
        current_prompt += " " + next_word
        print("current output: ")
        print(output)
        print("-----------------------------")

        # Check the length of output
        
        output_length = get_current_token_length(output)
        print(output_length)
    return output

"""
This function function checks if the initial prompt occurs
within n words of the end of the current prompt. If not
some text is deleted from the middle of the current prompt to meet this
condition.
"""
def adjust_prompt(prompt, initial_prompt, n):
    initial_num_words = len(initial_prompt.split())
    words = prompt.split()
    num_words = len(words)
    if num_words - initial_num_words <= n:
        return prompt
    else:
        words = words[:initial_num_words] + words[num_words - n:]
        prompt = ""
        for word in words:
            prompt += word + " "
        return prompt

if __name__ == "__main__":

    while(True):
        file_name = input("Enter a file with a prompt: ")
        with open(file_name, "r") as f:
            prompt = f.read()

        for i in range(3):
            print("Generating output", str(i), "for prompt:", prompt)
            print("-------------------------------------------------------")
            print("Output: ")
            
            output = generate_with_prompt(prompt)
            print(output)

            with open(file_name, "a", encoding="utf-8") as f:
                f.write("\n------------------------------------------------------\n")
                f.write("Output:\n")
                f.write(output)
                f.write("\n")
