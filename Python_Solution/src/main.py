from transformers import pipeline
import math

# using pre trained model = ez
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
    strs.append(string[init:init+512])
    init += 512

# run classifier on each string
strs = [clf(string) for string in strs]

"""the model handles embedding and all that fun stuff for us so I don't worry about it"""

# output our results
[print(x) for x in strs]

# get average sentiment
sentiment = []
for x in strs:
    if x[0]['label'] == 'NEGATIVE':
        sentiment.append(-1*x[0]['score'])
    else:
        sentiment.append(x[0]['score'])

print(sentiment)
sentiment = math.fsum(sentiment)

# output the sentiment of input.txt
print("NEGITIVE" if sentiment < 0 else "POSITIVE")
print(sentiment)
