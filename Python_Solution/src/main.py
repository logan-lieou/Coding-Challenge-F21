from transformers import pipeline
import math

# lazy moment

clf = pipeline("sentiment-analysis")

# read in the input file
with open('./input.txt', 'r') as f:
    string = f.read()
    f.close()

"""
the pipeline max input is 512
we want to ceil our input because
let's say we have a 516 long string
we still want to capture that 1 word
"""
x = math.ceil(len(string) / 512)

"""
create a list of strings from the input where
we make a new string every 512
this is really scuffed ngl
"""
init = 0
strs = []

for i in range(x):
    strs.append(string[:init+512])

# run classifier on each string
strs = [clf(string) for string in strs]

# get average label
label = [x[0]['label'] for x in strs]
label = list(map(lambda x: 1 if x=='POSITIVE' else 0, label))
label = round(math.fsum(label) / len(label))

# get average sentiment
sentiment = [x[0]['score'] for x in strs]
sentiment = math.fsum(sentiment) / len(sentiment)


# output the sentiment of the input
print("NEGITIVE" if label == 0 else "POSITIVE")
print(sentiment)
