using Flux
using Flux: onehot
using Flux.Optimise: update!
using Transformers
using WordTokenizers
using Transformers.Basic
using BSON: @save

include("data.jl")

vocab = Vocabulary(labels, unksym)

const embed = Embed(512, length(vocab); scale=inv(sqrt(512)))

const encoder = todevice(Stack(
    @nntopo(e → pe:(e, pe) → x → x → $N),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [Transformer(512, 8, 64, 2048) for i = 1:N]...
))

const decoder = todevice(Stack(
    @nntopo((e, m, mask):e → pe:(e, pe) → t → (t:(t, m, mask) → t:(t, m, mask)) → $N:t → c),
    PositionEmbedding(512),
    (e, pe) -> e .+ pe,
    Dropout(0.1),
    [TransformerDecoder(512, 8, 64, 2048) for i = 1:N]...,
    Positionwise(Dense(512, length(labels)), logsoftmax)
))

const ps = params(embed, encoder, decoder)
const opt = ADAMW(lr)

function smooth(et)
   global sMooth
   sm = fill!(similar(et, Float32), sMooth/size(embed, 2))
   p = sm .* (1 .+ -et)
   label = p .+ et .* (1 - convert(Float32, sMooth))
   label
end

Flux.@nograd smooth

function loss(m, src, trg, src_mask, trg_mask)
   lab = onehot(vocab, trg)

   src = m.embed(src)
   trg = m.embed(trg)

   if src_mask === nothing || trg_mask === nothing
       mask = nothing
   else
       mask = getmask(src_mask, trg_mask)
   end

   enc = m.encoder(src)
   dec = m.decoder(trg, enc, mask)

   #label smoothing
   label = smooth(lab)[:, 2:end, :]

   if mask === nothing
       loss = logkldivergence(label, dec[:, 1:end-1, :])
   else
       loss = logkldivergence(label, dec[:, 1:end-1, :], trg_mask[:, 1:end-1, :])
   end
end

function translate(x)
   ix = todevice(vocab(mkline(x)))
   seq = [startsym]

   src = embed(ix)
   enc = encoder(src)

   len = length(ix)
   for i = 1:2len
       trg = embed(todevice(vocab(seq)))
       dec = decoder(trg, enc, nothing)
       #move back to gpu due to argmax wrong result on CuArrays
       ntok = onecold(collect(dec), labels)
       push!(seq, ntok[end])
       ntok[end] == endsym && break
   end
   seq
end

function train!()
   global batch
   println("start training")
   i = 1
   model = (embed=embed, encoder=encoder, decoder=decoder)
   while (bAtch = get_batch(datas, batch)) != []
     x, t, x_mask, t_mask = preprocess(bAtch)
     grad = gradient(ps) do
         loss(model, x, t, x_mask, t_mask)
     end
     i+=1
     i%8 == 0 && @show loss(model, x, t, x_mask, t_mask)

     @time update!(opt, ps, grad)
   end
end

train!()

@save "transformer.bson" model