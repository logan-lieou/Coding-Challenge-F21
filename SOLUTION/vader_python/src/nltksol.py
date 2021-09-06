# from nltk import tokenize
import vader
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer

with open("../input.txt", "r") as f:
    ss = f.read()
    f.close()

csid = vader.SentimentIntensityAnalyzer()
sid = SentimentIntensityAnalyzer()
polarity, senti = csid.polarity_scores(ss)

# hacked
x = math.fsum(senti)/len(senti)
senti = x/math.sqrt(x+15)

print(f"overall sentiment: {round(senti, 4)}")
print(polarity)
print(sid.polarity_scores(ss))
