using TextAnalysis, Clustering
using Transformers, Transformers.Pretrain, Transformers.Basic
using Transformers.Basic: Vocabulary


fd = FileDocument("input.txt")
sd = StringDocument(text(fd))

remove_corrupt_utf8!(sd)
prepare!(sd, strip_punctuation)
stem!(sd)

bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

someText = text(sd) |> tokenizer |> wordpiece
someText = ["[CLS]"; someText; "[SEP]"]

token_indicies = vocab(someText)
segment_indices = [fill(1, length(someText)+2)]

data = (tok = token_indicies, segment = segment_indices)

"""
bert_embedding = sample |> bert_model.embed
feature_tensors = bert_embedding |> bert_model.transformers
"""

@show bert_model.embed(data)