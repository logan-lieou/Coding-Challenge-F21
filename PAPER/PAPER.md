## 2. VADER

The pretrained model thinks `{'neg': 0.065, 'neu': 0.748, 'pos': 0.187, 'compound': 0.9982}`
this means 6.5% of the text has a negative valence score 74% has a neutral valence score and 18% has a positive valence score. The model seems to believe that the text is mostly neutral in sentiment. Valence score is supposed to be representative of the strength of a particular sentiment negative scores i.e. x < 0 mean that the text has negative sentiment and positive scores x > 0 mean the text has positive sentiment.

### 2.1. Compound Score
On the compound score... "compound score" on the vaderSentiment github page it says the following.
> positive sentiment: compound score >= 0.05<br>
neutral sentiment: (0.05 < compound score > -0.05)<br>
negative sentiment: compound score <= -0.05<br>

After review of the source code this is not what compound score is and or this is misrepersentitive of what compound score is / how it is calculated. The way that compound score is calculated, is the sum of positive and negative valence scores normalized between -1 and +1.

Keep in mind positive and negative, not nuetral. So in the case of our input we run into an issue. Nuetral scores are totally ignored by the compound score. This is because it's a sum and nuetral scores are repersented as 0.0 having no weighting in the sum, so when we calculate compound score it's going to be way off the sentiment of the paper due to the absence of nuetral weighting, say I have a paper that is 99% neutral and 1% positive if this is the case the compound score will be the normalized valence score of just that 1% positive part of the input, not repersentitive of the input as a whole.

### 2.2. The Workaround
In order to get around the compound score issue I forked vader so that I could write my own compound score that weighted neutral sentiment, the way I weighted neutral sentiment is by averaging the values insteading of summing them, by averaging the values you give neutral scores weighting in the overall sentiment of the input. Then normalizing them, this new output would be more repersentitive of the input's sentiment than before. This new number is 0.025 which is within 0.05 of 0.0 repersenting a neutral score that tends slightly positive sentiment.