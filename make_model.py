
#copy the code and use it on google colab 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle
from google.colab import drive

drive.mount('/content/drive')
warnings.filterwarnings('ignore')
sns.set()

imdb=pd.read_csv("/content/drive/MyDrive/IMDB Dataset.csv")
imdb.head()

text= imdb['review'][0]
print(text)
print("<============>")
print(word_tokenize(text))


corpus=[]
for text in imdb["review"]:
    words=[word.lower() for word in word_tokenize(text)]
    corpus.append(words)

num_words = len(corpus)
print(num_words)

imdb.shape

train_size = int(imdb.shape[0]*0.8)
x_train = imdb.review[:train_size]
y_train = imdb.sentiment[:train_size]
x_test = imdb.review[train_size:]
y_test = imdb.review[train_size:]

tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(x_train)
x_train=tokenizer.texts_to_sequences(x_train)

x_train = pad_sequences(x_train,truncating= "post",maxlen=128,padding="post")

x_train[0],len(x_train[0])
x_test=tokenizer.texts_to_sequences(x_test)

x_test = pad_sequences(x_test,truncating= "post",maxlen=128,padding="post")

x_test[0],len(x_test[0])
le = LabelEncoder()
y_train = le.fit_transform(y_train)

y_test = le.fit_transform(y_test)

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=100, input_length=128, trainable = True))
model.add(LSTM(100,dropout=0.1,return_sequences=True))
model.add(LSTM(100,dropout=0.1))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history  = model.fit(x_train,y_train,epochs=5,batch_size=64,validation_data=(x_test,y_test))
validation_sentence = ['I appreciate it greatly']
validation_sentence_tokened = tokenizer.texts_to_sequences(validation_sentence)
validation_sentence_padded = pad_sequences(validation_sentence_tokened, maxlen=128, truncating='post', padding='post')
print(validation_sentence[0])

print("Positivity Index: ", model.predict(validation_sentence_padded)[0][0])
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("saved")
