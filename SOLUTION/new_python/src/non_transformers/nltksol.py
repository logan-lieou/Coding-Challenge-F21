from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

with open("../input.txt", "r") as f:
    ss = f.read()
    f.close()

sid = SentimentIntensityAnalyzer()
print(sid.polarity_scores(ss))


print(sid.polarity_scores("cock bitch ass motherfucker eat shit pussy boy"))
