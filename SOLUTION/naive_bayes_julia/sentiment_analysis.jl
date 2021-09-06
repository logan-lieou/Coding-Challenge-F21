using CSV, DataFrames, TextAnalysis
using TextAnalysis: NaiveBayesClassifier, fit!, predict
using BSON: @save

using Flux: tanh

df = DataFrame(CSV.File("data/data.csv"))

function preprocess_tweets(df)
   return df.text, df.airline_sentiment, unique(df.airline_sentiment)
end

function preprocess_sst2(df)
   df = df[:, ["body_text", "sentiment values"]]
	rename!(df, ["text", "sentiment"])
   sentiments = map((x) -> x >= 0.5 ? "negative" : "positive", df.sentiment)
   return df.text, sentiments, unique(sentiments)
end

texts, labels, uniques = preprocess_sst2(df)

function prepare_string_doc(s)
	sd = StringDocument(s)
	stem!(sd)
	return sd
end

model = let
	nbc = NaiveBayesClassifier(uniques)
	for (text, label) in zip(texts, labels)
		sd = prepare_string_doc(text)
		fit!(nbc, text, label)
	end
	nbc
end

function prediction(model, strings)
	doc = TextAnalysis.text.(prepare_string_doc.(strings))
	ana = predict.(Ref(model), doc)
	return ana
end

f = open("input.txt", "r")
pred_str = readlines(f)
close(f)
preds = prediction(model, pred_str)

@show preds

poss = [x["positive"] for x in preds]
poss = sum(poss)

negs = [x["negative"] for x in preds]
negs = -sum(negs)

@show tanh(negs + poss)
@show prediction(model, "I loved the flight")
@save "naivebayes.bson" model