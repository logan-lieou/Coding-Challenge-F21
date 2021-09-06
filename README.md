# ACM Research Coding Challenge (Fall 2021)

## 1. Original Approach
In the Python solution the approach was originally going to be just to take the entire input.txt file and feed it into a pre-trained transformer. But unfortunately that didn't work. So on the second try I decided to split up the contents of input.txt and pass parts of the file to the model one at a time. This is problematic due to the way Transformers work, Transformers are highly reliant on context, and due to this process of splitting up the file I am denying that context to the transformer thus having to process each string individually giving less accurate results.

The next problem we face with this approach is how to interpret the outputs in a way that is valid, the output of distilbert is the probability that the input matches a label this is basically the model’s confidence in its prediction i.e. an output of 0.99 is very confident and an output of 0.5 is the model being unsure. Attached to the confidence is a label, or the prediction of of whether the sentiment is positive or negative this is why the lower bound is 0.5 because after a probability of 0.5 it will just swap to predicting the other option. Throughout the input we see that the model fluctuates between positive and negative predictions. The way I get the overall sentiment of the input is by adding all positive predictions to all negative predictions, where positive predictions are positive and negative predictions are negative, the result will give me which predictions the model has more of.

### 1.1. Numbers
Using the approach I previously discussed we obtain the number 0.022 meaning that the model is tending slightly towards it’s positive predictions. Since each prediction will be positive or negative the sum could be very positive or very negative, actually there is no bound to how positive or negative a result can be therefore we need to squish the result, I do this using tanh if we have a input that is super, super positive say every sentence is 0.99 positive and there’s 100 sentences that means score is 99 by squishing this value with a tanh function we are able to find that this same value is just extremely close to 1.0 this process of squishing is called normalization and is commonly used to make unbounded outputs make sense. Passing the sum of output probabilities, to the tanh function we get 0.022 this normalized score is the exact same as our previous score, well it’s actually really close it’s not exactly 0.022 it’s more like 0.02199 but the score is very close to 0.022 so for convenience and rounding I will say that our normalized score is 0.022 based on this we can conclude that the sentiment of the input is slightly more positive than negative.

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

## 3. Naive Bayes
This is the solution that I was able to do in julia as implementing a transformer then training it was taking too long I decided to use Naive Bayes instead, this is built into a convient library in julia called TextAnalysis. Naive Bayes uses statistical infrence  in order to determine the probability of an event given another event. There's an entire paper on why Naive Bayes seems to work so well for sentiment analysis but I'm not going to go too much into it here basiclly know that Naive Bayes is really  really jank. Luckily our input seemed to have worked fairly well with the model I trained, although there are known issues with this model such as false positives and negatives.

### 3.1. Data and Cleaning
For the training I'm using the SST2 dataset
I produce a csv file for training with a python script called data.py that just appends the words to the sentiment scores. After that I read in the csv file to julia and do some magic.
```
function preprocess_sst2(df)
   df = df[:, ["body_text", "sentiment values"]]
	rename!(df, ["text", "sentiment"])
   sentiments = map((x) -> x >= 0.5 ? "negative" : "positive",
                    df.sentiment)
   return df.text, sentiments, unique(sentiments)
end
```
I used explore.jl to figure out what I needed to do to clean the data then used this function to get the things I needed for training the Naive Bayes model. There's another function in the file for cleaning a twitter dataset but the model was overwhelmingly negitive so I decided to use SST2 instead. What's going on here is that I select the columns I want, the text and sentiment columns, then I rename the columns for easy access. Next I map over the sentiment values and label each as either positive or negitive, we'll come back to this later. Lastly the function returns the text, sentiments (array of labels), and unique(sentiments). We can think of unique(sentiments) as being the A and B in, P(A) and P(B).

### 3.2. Training the Model
For training the model we have a simple function.
```
model = let
   nbc = NaiveBayesClassifier(uniques)
      for (text, label) in zip(texts, labels)
         sd = prepare_string_doc(text)
         fit!(nbc, text, label)
      end
   nbc
end
```
All the function is doing is taking in the possible events (A, B as mentioned before) then iterating over all the words and all the labels in our dataset.

### 3.3. Numbers
The output of the model is $0.995$ what does this mean? This means the model belives there is a $99.5\%$ probability the input is positive. How I got here here is that I summed all of the probabilities of positive and negitive events similar to how VADER works except instead of a valence score it's a probability in a particular classification. So you may be wondering how I delt with the nuetrality problem what if a score is nuetral say 0.5 which would be true nuetral for this model well the answer is jank you just ignore it, instead of saying 0.5 was true nuetral it's actually positive, and less than 0.5 is negitive this allows us to sum positive and negitive probabilities this model also relies on the idea that in any given input space there is less than alpha true nuetral sentiments, which should be true. Alpha in this case is repersentitive of arbitrary statistical confidence in a result. After we sum the probabilities we get some crazy unbounded value that is repersentitive of the probability that the sentiment is either positive or negitive in an unbounded distrobution. We normalize the value using tanh -1 being most negitive and +1 being most positive. Our output is then the probability of the input being either positive or negitive. * I HAVE NO IDEA IF THIS IS TRUE *

## 4. Naive Bayes vs VADER
You may be thinking what makes our score diffrent from valence score in VADER and why it's okay for this model to have an output like 0.995 positive but it's not okay for VADER to have that same output. Do the problems with VADER simply not apply to this model? So yes the problems with VADER do not apply here the reason is our model is outputting a probability where as VADER's valence score is outputting the strength of a particular sentiment. Let's say a word is +4 another word is -3 then the valence score is 1 a max valence score is 15 this score normalized is 0.25 what this means is that the sentiment  is decently positive. But with our model we take those two words as inputs and output the probability that it's positive and the probability that it's negitive in this case we may get something like "positive" => 0.99, "negitive" => 0.01, meaning that our model predicts that the sentence has a 99% chance of being positive.

## 5. Conclusion
The best solution was probably the transformer, but unfortuntely I had no time. Based on the other results of Naive Bayes, VADER, and dilbert I can conclude that input.txt file is mostly nuetral with a tendency towards positive sentiment. After reading the input myself I agree with the models the begining of the file is nuetral/positive then there's one part that could be negitive near the begining middle then after that the rest seems to be nuetral/positive.

## 6. Works Cited
- [1] C.J. Hutto and Eric Gilbert,  Vader Source Code.
- [2] Guillaume Klein and Yoon Kim and Yuntian Deng and Jean Senellart and Alexander M. Rush, The Annotated Transformer
- [3] H. Zhang (2004). The optimality of Naive Bayes. Proc. FLAIRS.