using Transformers.Datasets
using Transformers.Datasets: WMT, IWSLT

const N = 6
const sMooth = 0.4
const batch = 8
const lr = 1e-4
const max_len = 1000

data = WMT.GoogleWMT()
datas = dataset(Train, data)
vocab = get_vocab(data)

startsym = "<s>"
endsym = "</s>"
unksym = "</unk>"
labels = [unksym, startsym, endsym, collect(keys(vocab))...]

# getmask is really convinent
function preprocess(batch)
   x = mkline.(batch[1])
   t = mkline.(batch[2])
   x_mask = getmask(x)
   t_mask = getmask(t)
   x, t = vocab(x, t)
   x, t, x_mask, t_mask
end

function mkline(x)
   global max_len 
   xi = split(x)
   if length(xi) > max_len
     xi = xi[1:100]
   end

   [startsym, xi..., endsym]
end