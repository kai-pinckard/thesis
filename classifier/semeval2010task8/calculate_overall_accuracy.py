import json

with open("graded_step1_test_data.json", "r") as f:
    step1 = json.load(f)
with open("graded_step2_test_data.json", "r") as f:
    step2 = json.load(f)
with open("graded_step3_test_data.json", "r") as f:
    step3 = json.load(f)



def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

causal_only1 = []

for d in step1:
    if d["relation_type"] == 0:
        causal_only1.append(d)
    


causal_only2 = []

for d in step2:
    if d["relation_type"] == 0:
        causal_only2.append(d)


print(len(causal_only1))
print(len(causal_only2))
print(len(step3))


correct1 = []
for d in causal_only1:
    if d["correctly_predicted"] == 1:
        correct1.append(d)


correct2 = []
for d in causal_only2:
    if d["correctly_predicted"] == 1:
        correct2.append(d)

correct3 = []
for d in step3:
    if d["correctly_predicted"] == 1:
        correct3.append(d)


second_two_correct = []
def same_e1_e2(e1, e2, e21, e22):
    return e21 == e1 and e22 == e2

maxn = 0
minn = 1000
for i, d in enumerate(correct2):
    num = 0
    for j, d2 in enumerate(correct3):

        if same_e1_e2(d["e1_contents"], d["e2_contents"], d2["e1_contents"], d2["e2_contents"]):
            num += 1
            second_two_correct.append(d["e1_contents"] + d["e2_contents"])
    if num == 2:
        print(d)
    
    if num > maxn:
        maxn = num

    if num == 0:
        print(d)

    if minn > num:
        minn = num

print(maxn)
# this reveals that only there are only two causal sentences with identical e1 e2 tags. These will be handled manually in calculating the results
print(minn)
#this pair of e1 e2 labels matches another pair in the other dataset


print("second two", len(unique(second_two_correct)) + 1) # +1 for the reason described below

def all_match(e1, e2, e21, e22, e31, e32):
     return e21 == e1 and e22 == e2 and e31 == e21 and e32 == e22



print("correct lengths")
print(len(correct1))
print(len(correct2))
print(len(correct3))


all_correct = []
all_correct_e1_e2 = []
maxn = 0
minn = 1000
for i, d in enumerate(correct1):
    num = 0
    for j, d2 in enumerate(correct2):
        for v, d3 in enumerate(correct3):
            if all_match(d["e1_contents"], d["e2_contents"], d2["e1_contents"], d2["e2_contents"], d3["e1_contents"], d3["e2_contents"]):
                num += 1
                all_correct.append([d, d2, d3])
                all_correct_e1_e2.append(d["e1_contents"] + d["e2_contents"])
    if num == 2:
        print(d)
    
    if num > maxn:
        maxn = num

    if minn > num:
        minn = num
print(maxn)
# this reveals that only there are only two causal sentences with identical e1 e2 tags. These will be handled manually in calculating the results
print(minn)
#this pair of e1 e2 labels matches another pair in the other dataset

print(len(all_correct))

print(len(all_correct_e1_e2))


#the number of unique e1 e2 pairs that were correct for all steps + 1 equals the number of sentences that were correctly handled in all three steps. 
# the + 1 comes from the fact that there are two sentences with the same e1 e2 tags and both are correct on all three tasks however unique will remove one of them.
print("num:", len(unique(all_correct_e1_e2)) + 1)
print(unique(all_correct_e1_e2))
     