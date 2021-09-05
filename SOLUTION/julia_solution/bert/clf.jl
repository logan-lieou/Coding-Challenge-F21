using Flux
using Flux: onecold, onehot
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using Transformers.Datasets
using Transformers.BidirectionalEncoder

f = open("input.txt", "r")
lines = read(f)
close(f)

const labels = unique(lines)

markline(sent) = ["[CLS]"; sent; "[SEP]"]
function preprocess(batch)
    sentence = markline.(wordpiece.(tokenizer.(batch[1])))
    mask = getmask(sentence)
    tok = vocab(sentence)
    segment = fill!(similar(tok), 1)

    label = onehotbatch(batch[2], labels)
    return (tok=tok, segment=segment), label, mask
end

const _bert_model, wordpiece, tokenizer = pretrain"Bert-uncased_L-12_H-768_A-12"
const vocab = Vocabulary(wordpiece)

const hidden_size = size(_bert_model.classifier.pooler.W ,1)
const clf = gpu(Chain(
    Dropout(0.1),
    Dense(hidden_size, length(labels)),
    logsoftmax
))

const bert_model = gpu(set_classifier(_bert_model,(pooler = _bert_model.classifier.pooler, clf = clf)))

const ps = params(bert_model)
const opt = ADAM(1e-6)

@show ps

function acc(p, label)
    pred = Flux.onecold(collect(p))
    label = Flux.onecold(collect(label))
    sum(pred .== label) / length(label)
end


function loss(data, label, mask=nothing)
    e = bert_model.embed(data)
    t = bert_model.transformers(e, mask)

    p = bert_model.classifier.clf(
        bert_model.classifier.pooler(
            t[:,1,:]
        )
    )

    l = Basic.logcrossentropy(label, p)
    return l, p
end
@show loss(lines, unique(lines))
