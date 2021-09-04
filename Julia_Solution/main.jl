using Flux
using Transformers
using Transformers.Basic

labels = collect(1:10)
startsym = 11
endsym = 10
unksym = 0
labels = [unksym, startsym, endsym, labels...]
vocab = Vocabulary(labels, unksym)

# generate training data real fast
sample_data() = (d = rand(1:10, 10); (d, d))

# function for adding start and ending symbol
preprocess(x) = [startsym, x..., endsym]

@show sample = preprocess.(sample_data())
@show encoded_sample = vocab(sample[1])

# defining the model
embed = Embed(1024, length(vocab))

# define a position embedding layer
pe = PositionEmbedding(1024)

# wrapper for get embedding
function embedding(x)
   we = embed(x, inv(sqrt(1024)))
   e = we .+ pe(we)
   return e
end

# define the 2 layer transformer
encode_t1 = Transformer(1024, 8, 64, 2048)
encode_t2 = Transformer(1024, 8, 64, 2048)

# define the 2 layer decoder
decode_t1 = TransformerDecoder(1024, 8, 64, 2048)
decode_t2 = TransformerDecoder(1024, 8, 64, 2048)

# define the layer to get the final output probability
linear = Positionwise(Dense(1024, length(vocab)), logsoftmax)

function encoder_forward(x)
   e = embedding(x)
   t1 = encode_t1(e)
   t2 = encode_t2(t1)
   return t2
end

function decoder_forward(x, m)
   e = embedding(x)
   t1 = decode_t1(e, m)
   t2 = decode_t2(t1, m)
   p = linear(t2)
   return p
end

enc = encoder_forward(encoded_sample)
probs = decoder_forward(encoded_sample, enc)

@show probs