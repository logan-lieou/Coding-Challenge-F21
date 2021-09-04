## Python Solution
Okay this is way more scuffed than I was originally anticipating, so essentially what's going on here?
I split the file up into sets of chars that can be run on the model the model is a distilbert model from
transformers. I run the model on each set of characters then average out the results. 
distilbert is a pretrained model I didn't train this model.

## Overall Sentiment Score
The overall score of the file is ~0.95 the label is negitive. This number repersents the confidence of the model
distilbert belives that there is a 95% chance that the file is a negitive sentiment
