import vader
import math

with open("input.txt", "r") as f:
    ss = f.read()
    f.close()

sid = vader.SentimentIntensityAnalyzer()
print(sid.polarity_scores(ss)) # will output modified compound score