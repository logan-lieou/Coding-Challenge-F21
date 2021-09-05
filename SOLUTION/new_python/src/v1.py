from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08,
                                                 clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalCrossentropy('accuracy')])

# data cleaning
features = pd.read_csv("data/features.csv")
labels = pd.read_csv("data/labels.csv")

features = features.drop(columns=['Unnamed: 0'])
labels = labels.drop(columns=['Unnamed: 0'])

num_rows = features.shape[0]
train_num = round(num_rows * 0.7)

train_x = features[0:train_num]
test_x = features[train_num:num_rows]

train_y = labels[0:train_num]
test_y = labels[train_num:num_rows]

# train the model
model.fit(train_x, train_y, epochs=10)

tf_batch = tokenizer(strs, max_length=128, padding=True, trucation=True, return_tensors='tf')
