import json
from transformers import AlbertTokenizer, AlbertForMaskedLM, pipeline
import datetime


models = [
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2"
]

patterns = [
            {
                "prompt": "Did [CAUSE] cause [EFFECT]? [MASK].",
                "isaffirmative": True,
            },
            {
                "prompt": "Did [EFFECT] cause [CAUSE]? [MASK].",
                "isaffirmative": False,
            },
            {
                "prompt": "Was [EFFECT] the result of [CAUSE]? [MASK].",
                "isaffirmative": True,
            },
            {
                "prompt": "Was [CAUSE] the result of [EFFECT]? [MASK].",
                "isaffirmative": False,
            },
            {
                "prompt": "Was [CAUSE] the cause of [EFFECT]? [MASK].",
                "isaffirmative": True,
            },
            {
                "prompt": "If not for [CAUSE] would [EFFECT]? [MASK].",
                "isaffirmative": False,
            },
            {
                "prompt": "Is it true that [CAUSE] caused [EFFECT]? [MASK].",
                "isaffirmative": True,
            },
            {
                "prompt": "Is it true that [EFFECT] caused [CAUSE]? [MASK].",
                "isaffirmative": False,
            }
            #"For what reason did [EFFECT]?",
            #"What was the result of [CAUSE]",
            #"Is it true that, [CAUSE] caused [EFFECT]|yes",
            #"True or false, ..."
        ]
"""
example statement
statement = {}
statement["sent"] = "A causes B."
statement["cause"] = "A"
statement["effect"] = "B"
statement["iscausal"] = True
"""

def remove_period(text):
    if text.endswith("."):
        return text[:-1]
    return text

"""
This function takes a statement and a prompt as an input.
It uses the cause and effect from the statement to populate the prompt,
the prompt is then returned.
"""
def populate_prompt(statement, prompt):
    cause = remove_period(statement["cause"])
    effect = remove_period(statement["effect"])
    filled_prompt = prompt.replace("[CAUSE]", cause)
    filled_prompt = filled_prompt.replace("[EFFECT]", effect)
    return filled_prompt

"""
This function takes a statement and a populated prompt as an input.
It returns a question containing both.
"""
def make_question(statement, pattern):
    return statement["sent"] + ". " + populate_prompt(statement, pattern["prompt"])

"""
This function returns true if a question formed from the statement
and the prompt should have an affirmative answer. 
"""
def get_correct_answer(statement, pattern):
    if statement["iscausal"] and pattern["isaffirmative"]:
        return True
    else:
        return False

"""
This function takes as an input a list of dictionaries containing causal information.
Each dictionary represents a statement. A statement is a sentence that contains causal a causal relationship.
Thus, it has a cause and an effect. Thus, a statement has the attributes "sent", "cause", "effect"

A benchmark file is a json file containing a list of questions. Each question contains an informational sentence, a prompt
to get the model to answer these questions, the cause, the effect, and an expected answer.
"""
def create_benchmark_data(statements, patterns):
    questions = []
    for statement in statements:
        for pattern in patterns:
            question = {}
            question["question"] = make_question(statement, pattern)
            question["answer"] = get_correct_answer(statement, pattern)
            question["cause"] = statement["cause"]
            question["effect"] = statement["effect"]
            question["pattern"] = pattern["prompt"]
            questions.append(question)
    return questions

"""
This function takes a json statements file as input and outputs a benchmark_file
"""
def create_benchmark_file(statements_file, benchmark_file):
    with open(statements_file, "r") as f:
        statements = json.load(f)

    benchmark = create_benchmark_data(statements, patterns)
    
    with open(benchmark_file, "w") as f:
        json.dump(benchmark, f, indent=3)

"""
    This function is used by run_benchmark to check if the models replacement
    for mask should be considered correct.
"""
def is_correct(output, answer):
    if answer:
        fill_word = "yes"
    else:
        fill_word = "no"

    top_prediction = str(output[0]["token_str"][1:]).strip()
    return top_prediction == fill_word

"""
Iterates over the results for each pattern to calculate and return the
overall results
"""
def compute_overall_results(results):
    overall_result = {}
    overall_result["false_positives"] = 0
    overall_result["false_negatives"] = 0
    overall_result["total_questions"] = 0
    overall_result["correct"] = 0
    overall_result["accuracy"] = 0.0
    overall_result["pattern"] = "Overall Results"

    for result in results:
        overall_result["false_positives"] += result["false_positives"]
        overall_result["false_negatives"] += result["false_negatives"]
        overall_result["total_questions"] += result["total_questions"]
        overall_result["correct"] += result["correct"]

    overall_result["accuracy"] = float(overall_result["correct"])/overall_result["total_questions"]
    return overall_result

"""
This function evaluates a model on a benchmark_file
and stores the results in a results_file.
"""
def run_benchmark(model_name, benchmark_file, results_file, logging_file):
    with open(benchmark_file, "r") as f:
        benchmark = json.load(f)

    model = AlbertForMaskedLM.from_pretrained(model_name)
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    # Each pattern will store its own statistics
    results = []
    for pattern in patterns:
        result = {}
        result["false_positives"] = 0
        result["false_negatives"] = 0
        result["total_questions"] = 0
        result["correct"] = 0
        pattern["accuracy"] = 0.0
        result["pattern"] = pattern["prompt"]
        results.append(result)


    with open(logging_file, "w") as log:

        for benchmark_question in benchmark:

            output = fill_mask(benchmark_question["question"])

            output_str = output[0]["sequence"] + "\n"
            for o in output:
                output_str += str(o["token_str"][1:]) + " " + str(o["score"]) + "\n"
            print(output_str)
            log.write(output_str)

            # Update the correct patterns stats
            for result in results:
                if result["pattern"] == benchmark_question["pattern"]:
                    result["total_questions"] += 1
                    if is_correct(output, benchmark_question["answer"]):
                        result["correct"] += 1
                        print("correct")
                        log.write("correct\n")
                    else:
                        print("incorrect")
                        log.write("incorrect\n")
                        if benchmark_question["answer"] == True:
                            result["false_negatives"] += 1
                        else:
                            result["false_positives"] += 1
                    break

    # Calculate each pattern's accuracy
    for result in results:
        result["accuracy"] = float(result["correct"])/result["total_questions"]

    # Calculate and append the overall statistics
    results.append(compute_overall_results(results))
    results.append({"model_name": model_name, "datetime": str(datetime.datetime.now())})

    # Store the results -- downside of no results until the end.
    with open(results_file, "w") as f:
        json.dump(results, f, indent=3)

if __name__ == "__main__":
    create_benchmark_file("statements.json", "benchmark.json")
    run_benchmark("albert-base-v2", "benchmark.json", "results.json", "log.txt")