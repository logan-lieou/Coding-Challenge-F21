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

print(math.fsum(senti))
print(polarity)
print(sid.polarity_scores(ss))
