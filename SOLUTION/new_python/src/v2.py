from transformers import BertConfig, TFBertModel, BertTokenizer
import tensorflow as tf
import nltk

"""
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TFBertModel(BertConfig())

with open("input.txt", "r") as f:
    strs = f.read()
    f.close()

tokenized = tokenizer(strs, max_length=512, padding=True, truncation=True, return_tensors='tf')
res = model(tokenized)
print(res)

"""
tf_batch = tokenizer(strs, max_length=128, padding=True, truncation=True, return_tensors='tf')

tf_outputs = model(tf_batch)

predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ["POSITIVE", "NEGITIVE"]
label = tf.argmax(predictions, axis=1)
print(label)
"""
