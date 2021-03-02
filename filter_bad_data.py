import json

with open(".\\classifier\\semeval2010task8\\augmented_simplified_chinese.json", "r") as f:
    data = json.load(f)

filtered_data = []
for datapoint in data:
    if datapoint["sent"].find(datapoint["e1_contents"]) == -1:
        print(datapoint["sent"], "||||", datapoint["e1_contents"])
        continue
    if datapoint["sent"].find(datapoint["e2_contents"]) == -1:
        print(datapoint["sent"], "||||", datapoint["e2_contents"])
        continue
    if datapoint["relation_type"] == 9:
        # This means the relation type is not known
        continue
    if datapoint["e1_contents"].lower() == datapoint["e2_contents"].lower():
        continue

    filtered_data.append(datapoint)

print(len(filtered_data), len(data))
with open(".\\classifier\\semeval2010task8\\filtered_augmented_simplified_chinese.json", "w") as f:
    json.dump(filtered_data, f, indent=4)