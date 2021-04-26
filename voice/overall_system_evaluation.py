#!/usr/bin/env python
# coding: utf-8

import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[2]:

#################################################################
### Step 1
#################################################################
import json


from transformers import RobertaTokenizer

# Load the RoBERTa tokenizer.
print('Loading BERT tokenizer...')
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)



from transformers import RobertaForSequenceClassification, AdamW, RobertaConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = RobertaForSequenceClassification.from_pretrained(
    './step_1_casual_sentence_classifier_model', # use my stored model
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()

import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# In[18]:


import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Load the dataset
with open("../classifier/semeval2010task8/semeval_testV4.json", "r") as f:
    test_data = json.load(f)

# Report the number of sentences.
print("number of test sentences", len(test_data))

# Create sentence and label lists
sentences = sentences = [ item["sent"].lower() for item in test_data]

print(sentences[0])
test_labels = [ item["relation_type"] for item in test_data]
print(test_labels[0])
# convert all relation type labels other than 0 to 1. This is used to train a classifier that only distinguishes between causal
# and noncausal relations.
for i, test_label in enumerate(test_labels):
    if test_label == 0:
        continue
    else:
        test_labels[i] = 1
        
print(test_labels[0])


# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        truncation=True,
                        padding="max_length"
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
test_labels = torch.tensor(test_labels)


# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')




num_causal_sents = 0
for test_label in test_labels:
    if test_label == 0:
        num_causal_sents += 1
percent_causal = num_causal_sents / len(test_labels.tolist())

print("Positive samples:", num_causal_sents, percent_causal)


# In[27]:


from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
  # Calculate and store the coef for this batch.  
  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
  matthews_set.append(matthews)


# In[29]:


correct_1 = []
incorrect_2 = []
for i, prediction in enumerate(predictions):
    if prediction == true_labels[i]:
        correct.append([prediction, i])
    else:
        incorrect.append([prediction, i])

storage = {}
storage["correct"] = correct_1
storage["incorrect"] = incorrect_1

with open("step_1_results", "w") as f:
    json.dump(storage, f, indent=4)


# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)


num_correct = 0
for i, pred in enumerate(flat_predictions):
    if pred == flat_true_labels[i]:
        num_correct += 1
print("overall num correct", num_correct)
num_sents = len(flat_predictions)
print("num sents:", num_sents)
print("overall accuracy", num_correct/num_sents)

# calculate the accuracy for causal relations sents only
correct_causal = 0
for i, pred in enumerate(flat_predictions):
    if pred == flat_true_labels[i] and pred == 0:
        correct_causal += 1
        
print("num correct causal", correct_causal)
num_causal = num_causal_sents
print("num causal", num_causal)
print("accuracy for causal", correct_causal / num_causal)


# calculate the accuracy for non causal relation sents only
correct_noncausal = 0
for i, pred in enumerate(flat_predictions):
    if pred == flat_true_labels[i] and pred == 1:
        correct_noncausal += 1
        
print("num correct noncausal", correct_noncausal)
num_noncausal = num_sents - num_causal_sents
print("num noncausal", num_noncausal)
print("overall accuracy noncausal", correct_noncausal / num_noncausal)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('Total MCC: %.3f' % mcc)


"""
# saving the trained model to disk
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './step_1_casual_sentence_classifier_model'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
"""
