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

now this seems correct but the issue lies in nuetral sentiment because nuetral sentiment doesn't contribute to sum_s even slightly more positive or negitive input will cause the model to belive that the overall sentiment is very positive or very negitive due to vader's tendency to give sentences 0.0 true neutral valance scores. Because vader gives sentences 0.0 true neutral scores even if the overall sentiment of a paper is maybe +-0.05 vader will show that the sentiment of the input is something like -.99 or .99 in the case of our output we can see thatwe run into this issue here even though a sentence maybe +0.05 positive or -0.05the vast majority of the paper is not repersented in the final compound score.
