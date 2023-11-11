


from openai import OpenAI
import json

client = OpenAI(
  api_key='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',
)
client.models.list()




with open('ai data.json', 'r') as openfile:
 
    # Reading from json file
    data = json.load(openfile)

fd = []

for i in data:
    response = client.completions.create(
    model="gpt-3.5-turbo",
    prompt=i + "\n complete the sentence above and keep the word count within 100"
    )
    print(response)
    fd.append(response)


# l = []
# for i in data:
#     t = i.split(" ")
#     t = " ".join(t[:30])
#     l.append(t)

# json_object = json.dumps(l, indent=4)
 
# # Writing to sample.json
# with open("ai data.json", "w") as outfile:
#     outfile.write(json_object)

