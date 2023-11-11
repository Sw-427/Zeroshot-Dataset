import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
#import custom_datasets
from multiprocessing.pool import ThreadPool
import time



def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + 1 * 2)
    
    n_spans = int(n_spans)
    
    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - 1)
        search_end = min(len(tokens), end + 1)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

#t = "Furthermore, my adaptability and ability to collaborate well with others make me well-suited for the working environment, which involves classrooms and large lecture halls. I am capable of standing and walking around for extended periods during exams, even in tiered classrooms."
t = "I am very exited for this opportunity and cannnot wait to join you guys."
#t = "Focus on your emotional well-being by setting resolutions that promote mental health. This couh  ld involve practicing mindfulness, meditation, or gratitude, and seeking professional help if needed. A resolution like I will meditate for 10 minutes every day or I will seek therapy to address my anxiety can help improve your mental health."
#t = "December is coming and its the last month of the year. Everyone will gather with there family and spend the holidays in cozy cloths. The new year will bring new oppertunity and hopes for everyone."
#t = "The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed. Repair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water. Trains on the west coast mainline face disruption due to damage at the Lamington Viaduct. Many businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town. First Minister Nicola Sturgeon visited the area to inspect the damage. The waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare. Jeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit. However, she said more preventative work could have been carried out to ensure the retaining wall did not fail.  It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we're neglected or forgotten,  she said.  That may not be true but it is perhaps my perspective over the last few days.  Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?  Meanwhile, a flood alert remains in place across the Borders because of the constant rain. Peebles was badly hit by problems, sparking calls to introduce more defences in the area. Scottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs. The Labour Party's deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand. He said it was important to get the flood protection plan right but backed calls to speed up the process.  I was quite taken aback by the amount of damage that has been done,  he said.  Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses.  He said it was important that  immediate steps  were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans. Have you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled. Email us on selkirk.news@bbc.co.uk or dumfries@bbc.co.uk."
#t = "The full cost of damage in Newton Stewart, one of the areas worst affected by recent severe weather, is still being assessed. The region, located in Dumfries and Galloway, has borne the brunt of heavy rainfall, strong winds, and flooding, causing widespread devastation to homes, infrastructure, and communities. Emergency response teams and local authorities are working tirelessly to evaluate the extent of the damage and provide relief to those affected. In Hawick, another severely impacted area, repair work is currently underway. The town, situated in the Scottish Borders, has witnessed extensive damage to properties, roads, and essential services. Local contractors and volunteers are collaborating to address the immediate needs of the community, focusing on restoring critical infrastructure and supporting affected residents. Throughout Peeblesshire, many roads remain impassable due to floodwaters and debris, compounding the challenges faced by residents and emergency services. The local authorities are prioritizing the clearing and reopening of roads to facilitate access and aid in relief efforts. Additionally, temporary road diversions and alternative transportation solutions are being explored to ensure the mobility of people and resources. The aftermath of this severe weather event serves as a stark reminder of the importance of disaster preparedness and resilience in the face of extreme climate events. Communities are coming together to support one another during these trying times, highlighting the resilience of the people affected by the disaster. Local and national government agencies are coordinating resources and assistance, offering a glimmer of hope to those grappling with the aftermath. As assessments continue and repair work progresses, it is evident that a sustained and collaborative effort will be required to rebuild and rehabilitate the affected areas fully. The lessons learned from this disaster will inform future planning and infrastructure improvements to mitigate the impact of such events in the years to come. In the meantime, the indomitable spirit of the affected communities remains a source of inspiration, as they strive to recover and rebuild their lives in the wake of this natural disaster"   
masked_texts = []
for i in range(100):
    masked_texts.append(tokenize_and_mask(t,2,0.3))

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

mask_filling_model_name = "t5-large"
mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=512)
mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name)
       


pattern = re.compile(r"<extra_id_\d+>")


def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=0.9, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]
    print(tokens)
    n_expected = count_masks(masked_texts)
    print(n_expected)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

raw_fills = replace_masks(masked_texts)
extracted_fills = extract_fills(raw_fills)


perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)


print(perturbed_texts)

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load a pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other GPT-2 variants like "gpt2-medium", "gpt2-large", etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define p_theta as a function that calculates the log probability of a text passage
def p_theta(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids).logits
    # Calculate the log probability of the text passage
    log_prob = torch.log_softmax(logits, dim=-1)
    return log_prob[0, -1, input_ids[0, -1]]


def perturbation_discrepancy(x, p_theta, x_perturbed, num_samples=100):
    perturbation_discrepancies = []
    log_prob_original = p_theta(x)
    for i in x_perturbed:
        # Calculate the log probability of the original text and perturbed text
        
        log_prob_perturbed = p_theta(i)
        
        # Compute the perturbation discrepancy for this perturbation sample
        perturbation_discrepancy = log_prob_original / log_prob_perturbed
        perturbation_discrepancies.append(perturbation_discrepancy)
        
    # Calculate the expected perturbation discrepancy
    expected_discrepancy = (sum(perturbation_discrepancies))/num_samples
    
    return expected_discrepancy



# Example text passage

# Calculate the perturbation discrepancy
discrepancy = perturbation_discrepancy(t, p_theta, perturbed_texts)

print(discrepancy)

