from googletrans import Translator


translation_language = "es"

translator = Translator()

def translate_sentence_to(sent, lang):
    return translator.translate(sent, dest=lang)

def translate_sentence_from(sent, lang):
    return translator.translate(sent, src=lang, dest="en")

def reword_sentence(sent, lang):
    sent = translate_sentence_to(sent, lang)
    print(sent.text)
    sent = translate_sentence_from(sent.text, lang)
    print(sent.text)
    return sent.text

if __name__ == "__main__":

    import json
    with open(".\\classifier\\semeval2010task8\\semeval_datasetV2.json", "r") as f:
        dataset = json.load(f)

    print(len(dataset))
    dataset = dataset[:1]
    print(dataset)
    reworded_data = []
    translation_language = "es"


    for d in dataset:
        data_point = {}
        data_point["sent"] = reword_sentence(d["sent"], translation_language)
        data_point["e1_contents"] = reword_sentence(d["e1_contents"], translation_language)
        data_point["e2_contents"] = reword_sentence(d["e2_contents"], translation_language)
        data_point["is_active"] = d["is_active"]
        data_point["relation_type"] = d["relation_type"]
        data_point["translation_language"] = translation_language
        reworded_data.append(data_point)
    
    print(reworded_data)

    with open(".\\classifier\\semeval2010task8\\augmented.json", "w") as f:
        json.dump(reworded_data, f, indent=4)

    #print(reword_sentence("hi there", "es"))