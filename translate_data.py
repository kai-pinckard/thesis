from googletrans import Translator
import gensim.downloader as api
import re
print("loading word2vec")
wv = api.load('word2vec-google-news-300')
print("finished loading")
translation_language = "es"

translator_to_lang = Translator()
translator_to_eng = Translator()


def translate_sentence_to(sent, lang):
    return translator_to_lang.translate(sent, dest=lang)

def translate_sentence_from(sent, lang):
    # make a pointless request to interfere with translation caching
    _ = translator_to_eng.translate("please translate this sentence.")
    return translator_to_eng.translate(sent, src=lang, dest="en")

# rename to reword chunk
def reword_sentence(sent, lang):
    sent = translate_sentence_to(sent, lang)
    #print(sent.text)
    sent = translate_sentence_from(sent.text, lang)
    #print(sent.text)
    return sent.text

# uses word to vec to find the closest word to the e1/e2 contents when the translated words are not an exact match.
def get_closest_word(sent, word):
    sent = re.sub(r'[^\w\s]','', sent)

    best_match_index = -1
    best_match = -1
    sent_words = sent.split()

    for i, w in enumerate(sent_words):
        
        if w in wv.vocab:
            similarity = wv.similarity(word, w)
            print(word, w, similarity)
            if similarity >= best_match:
                best_match_index = i
                best_match = similarity
    print("e1: ", word, "best_match in sent", sent_words[best_match_index])
    return sent_words[best_match_index], best_match


# ensures that the e1 and e2 contents are exactly contained inside the sentence
def validate_translation(datapoint):

    # use this value to set the minimum threshold for similarity between e1 contents and the best match 
    # in the sentence. if the best match is less similar than the threshold the datapoint will not be kept.
    # this is not yet implemented
    SIMILARITY_THRESHOLD = 0.1

    sent = datapoint["sent"]
    e1 = datapoint["e1_contents"]
    e2 = datapoint["e2_contents"]

    # This is a bad translation and it can not be fixed
    discard = False
    try:
        if sent.find(e1) == -1:
            if len(e1.split()) > 1:
                discard = True
            else:
                closest_word, match_quality = get_closest_word(sent, e1)
                datapoint["e1_contents"] = closest_word
        if sent.find(e2) == -1:
            if len(e2.split()) > 1:
                discard = True
            else:
                closest_word, match_quality = get_closest_word(sent, e2)
                datapoint["e2_contents"] = closest_word
    except:
        discard = True
    return datapoint, discard
  
if __name__ == "__main__":

    import json
    with open(".\\classifier\\semeval2010task8\\semeval_datasetV2.json", "r") as f:
        dataset = json.load(f)

    #print(len(dataset))
    dataset = dataset[:10]
    #print(dataset)
    reworded_data = []

    # simplified chinese lang code = zh-CN
    # spanish lang code = es
    translation_language = "zh-CN"#"es"

    # temporarily change sent to *** so it will not be translated
    for i, datapoint in enumerate(dataset):
        dataset[i]["1"] = datapoint["sent"]
        dataset[i].pop("sent")
        dataset[i]["2"] = datapoint["relation_type"]
        dataset[i].pop("relation_type")
        dataset[i]["3"] = datapoint["is_active"]
        dataset[i].pop("is_active")
        dataset[i]["4"] = datapoint["e1_contents"]
        dataset[i].pop("e1_contents")
        dataset[i]["5"] = datapoint["e2_contents"]
        dataset[i].pop("e2_contents")

    # split the json file into chunks
    data_chunks = []
    data_chunk = "" 
    chunk_num = 0
    for i, data_point in enumerate(dataset):
        # split the json data into chunks of size 50 to be below the 15,000 char limit
        if i % 25 == 0 and i != 0:
            data_chunks.append(data_chunk)
            data_chunk = ""
        data_point_str = json.dumps(data_point)
        data_chunk += data_point_str + ",\n"
    data_chunks.append(data_chunk)

    # remove all characters after the last } for proper json formatting
    for i, data_chunk in enumerate(data_chunks):
        data_chunks[i] = data_chunks[i][:data_chunk.rfind("}") + 1]

    print(data_chunks)
    reworded_data = []
    for chunk in data_chunks:
        #st = "[\n" + reword_sentence(chunk, translation_language) + "\n]"
        #print(st)
        #print(st[216])
        #json.loads(st)
        json_text = "[\n" + reword_sentence(chunk, translation_language) + "\n]"
        json_text = json_text.replace(" \"", "\"")
        json_text = json_text.replace("\" ", "\"")
        json_text = json_text.replace(" ,", ",")
        print("++", json_text)
        try:
            reworded_data.append( json.loads(json_text) )
        except:
            continue
    print("-----------------------------------------------")
    print(reworded_data)
    with open("raw_text.txt", "w") as f:
        f.write(str(reworded_data))

    #flatten the list
    flattened_data = []
    for data_list in reworded_data:
        for data in data_list:
            flattened_data.append(data)

    print("===================",flattened_data)
    # convert temporary keys back to old keys
    for i, datapoint in enumerate(flattened_data):
        flattened_data[i]["sent"] = flattened_data[i]["1"]
        flattened_data[i].pop("1")
        flattened_data[i]["relation_type"] = flattened_data[i]["2"]
        flattened_data[i].pop("2")
        flattened_data[i]["is_active"] = flattened_data[i]["3"]
        flattened_data[i].pop("3")
        print("---------------------",flattened_data[i])
        flattened_data[i]["e1_contents"] = flattened_data[i]["4"]
        flattened_data[i].pop("4")
        flattened_data[i]["e2_contents"] = flattened_data[i]["5"]
        flattened_data[i].pop("5")


    omitted = []
    print("-------------")
    for i, datapoint in enumerate(flattened_data):
        print(datapoint)
        flattened_data[i], should_discard = validate_translation(datapoint)
        if should_discard is True:
            omitted.append(flattened_data[i])
            del flattened_data[i]
    print("-----------------------------------------------")
    print(omitted)
    print("generated", len(flattened_data), "datapoints")

    with open(".\\classifier\\semeval2010task8\\augmented.json", "w") as f:
        json.dump(flattened_data, f, indent=4)

    #print(reword_sentence("hi there", "es"))