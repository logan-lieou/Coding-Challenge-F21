using Languages
using Unicode
using CSV
using DataFrames
using Flux

df = DataFrame(CSV.File("data.csv"))

"""
list_stopwords = stopwords(Languages.English())

# data cleaning #
function remove_stopwords(tokens)
   filtered_sentence = []
   for token in tokens
      if !(lowercase(token) in list_stopwords)
         push!(filtered_sentence, lowercase(token))
      end
   end

   return filtered_sentence
end

function convert_clean(arr)
   arr = string.(arr)
   arr = Unicode.normalize.(arr, stripmark=true)
   arr = map(x -> replace(x, r"[^a-zA-Z0-9_]" => ""), arr)
   return arr
end
"""

model = Chain(
   LSTM(1024, 512),
   LSTM(512, 256),
   LSTM(256, 128),
   LSTM(128, 10),
   Dense(10, 2),
   softmax)

L(x, y) = Flux.mse(model(x), y)
opt = ADAMW(0.02)

# labels = df[:, "labels"]

@show head(df)
