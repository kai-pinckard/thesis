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

"""
this works! Note that you need to remove all punctuation from the words otherwise it will
crash by saying that the word is not in the vocabulary.
from gensim.models import Word2Vec

import gensim.downloader as api
wv = api.load('word2vec-google-news-300')


sente = "the baby cried in its crib"
for word in sente.split():
    print(word, "cradle")
    print(wv.similarity(word, "cradle"))
"""

if __name__ == "__main__":

    import json
    with open(".\\classifier\\semeval2010task8\\semeval_datasetV2.json", "r") as f:
        dataset = json.load(f)

    #print(len(dataset))
    dataset = dataset[:2]
    #print(dataset)
    reworded_data = []
    translation_language = "es"

    # split the json file into chunks
    data_chunks = []
    data_chunk = "" 
    chunk_num = 0
    for i, data_point in enumerate(dataset):
        # split the json data into chunks of size 50 to be below the 15,000 char limit
        if i % 50 == 0 and i != 0:
            data_chunks.append(data_chunk)
            data_chunk = ""
        data_point_str = json.dumps(data_point)
        data_chunk += data_point_str + ",\n"

    if data_chunk.endswith(","):
        data_chunk = data_chunk[:-1]
    data_chunks.append(data_chunk)

    #print(data_chunks)
    reworded_data = []
    for chunk in data_chunks:
        reworded_data.append(json.loads(reword_sentence(chunk, translation_language)))


    #print(reworded_data)
    with open(".\\classifier\\semeval2010task8\\augmented.json", "w") as f:
        json.dump(reworded_data, f, indent=4)

    #print(reword_sentence("hi there", "es"))