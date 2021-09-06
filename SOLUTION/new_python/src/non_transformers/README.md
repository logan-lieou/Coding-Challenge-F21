## NLTK Solution

The pretrained model thinks `{'neg': 0.065, 'neu': 0.748, 'pos': 0.187, 'compound': 0.9982}`

this means 6.5% of the text has a negitive valence score 74% has a nuetral valence score and 18% has a positive valence score.

the model seems to belive that the text is mostly nuetral

now the issue here lies in "compound score"

```
def _sift_sentiment_scores(self, sentiments):
   # want separate positive versus negative sentiment scores
   pos_sum = 0.0
   neg_sum = 0.0
   neu_count = 0
   for sentiment_score in sentiments:
      if sentiment_score > 0:
            pos_sum += (
               float(sentiment_score) + 1
            )  # compensates for neutral words that are counted as 1
      if sentiment_score < 0:
            neg_sum += (
               float(sentiment_score) - 1
            )  # when used with math.fabs(), compensates for neutrals
      if sentiment_score == 0:
            neu_count += 1
   return pos_sum, neg_sum, neu_count
```

after review the source code of vader I found that vader = probably bugged, so for this project I needed to fork vader to modify the
source code in order to access encapsulated variables so that I could discover what was going on (needed to get sentiment) what I found is what I already thought from reviewing the above code.

On the vaderSentiment github page [here](https://github.com/cjhutto/vaderSentiment#about-the-scoring) it says:

```
positive sentiment: compound score >= 0.05
neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
negative sentiment: compound score <= -0.05
```

this is incorrect the compound score is NOT repersentitive of sentiment, this is due to the fact that a nuetral sentiment contributes 0 to the sum_s meaning that even if text is mostly nuetral the score will reflect the CONFIDENCE OF VADER IN THE TENDENCY of the input not weather or not the input is positive negitive or neutral.


Example:
18% of the input is negitive 3% is positive and 79% is neutral you would expect the compound score to reflect the sentiment of the input as neutral as claimed by the maintainers. But what will actually happen is vader does this:

$$
\frac{sum_s}{\sqrt{sum_s+\alpha}}, sum_s = \Sigma{}p_i n_i
$$

the problem is that it will give a normalized sum normalized to a scale of [-1, 1] 0 being true nuetral if there's the same amount of positive and negitive words then the score will end up at 0, the problem here is obvious with the way vader works the model is context independent meaning that although words may be positive they can be used in a negitive way. Vader is notoriously bad at detecting sarcasm. a way you may be able to get around this is through the normalized mean instead of using normalized sum the issue with normalized sum is that nuetral paragraphs are completely ignored even if the majority of the input is nuetral.

The main issue here is valaence score is supposed to indicate how positive or negitive something is +4, -4 etc. so a normalized score would imply that the input is 99% positive when in all reality the score should be something like 15% positive the score is overall not repersentitive of the sentiment of the input ja?