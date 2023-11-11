

import requests
import json

# Define the API endpoint URL
url = "https://datasets-server.huggingface.co/rows?dataset=EdinburghNLP%2Fxsum&config=default&split=train&offset=0&length=100"

# Make a GET request to the API
response = requests.get(url)

import re
# Emoji removing pattern
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Emojis in the Emoticons block
                           u"\U0001F300-\U0001F5FF"  # Emojis in the Miscellaneous Symbols and Pictographs block
                           u"\U0001F700-\U0001F77F"  # Emojis in the Alchemical Symbols block
                           u"\U0001F780-\U0001F7FF"  # Emojis in the Geometric Shapes Extended block
                           u"\U0001F800-\U0001F8FF"  # Emojis in the Supplemental Arrows-C block
                           u"\U0001F900-\U0001F9FF"  # Emojis in the Supplemental Symbols and Pictographs block
                           u"\U0001FA00-\U0001FA6F"  # Emojis in the Chess Symbols block
                           u"\U0001FA70-\U0001FAFF"  # Emojis in the Symbols and Pictographs Extended-A block
                           u"\U0001F004-\U0001F0CF"  # Additional emoticons
                           u"\U0001F170-\U0001F251"  # Additional emoticons
                           "]+", flags=re.UNICODE)


pattern = r'[^\x00-\x7F]+'
data = response.json()

def rb(text):
    # Remove URLs
    text_without_links = re.sub(r'http\S+', '', text)
    
    # Remove brackets and content inside them
    text_without_brackets = re.sub(r'\{.*?\}|\(.*?\)', '', text_without_links)    
    text_without_brackets = re.sub(r'[\[\]{}()]', '', text_without_brackets)
    text_without_brackets = re.sub(r'\(.*?\)', '', text_without_brackets)
    # Remove Emojis
    text_without_brackets = emoji_pattern.sub(r'', text_without_brackets)
    # Remove emojis
    text_without_brackets = re.sub(pattern, '', text_without_brackets)
    # Remove Extra Spaces
    cleaned_text = re.sub(r'\s+', ' ', text_without_brackets)

    return cleaned_text.strip()

l = []

for i in data["rows"]:
    if i["row_idx"]<= 500:
        l.append(rb(i["row"]["document"]))

print(len(l))

json_object = json.dumps(l, indent=4)
 
# Writing to sample.json
with open("human data.json", "w") as outfile:
    outfile.write(json_object)
