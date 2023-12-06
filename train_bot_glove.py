import json 
import numpy as np 
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, SimpleRNN,Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
#print(stop_words)
with open('C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []



for intent in data['intents']:
    for pattern in intent['patterns']:
        words = word_tokenize(pattern.lower())
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_sentence = ' '.join(filtered_words)
        training_sentences.append(filtered_sentence)
        training_labels.append(intent['tag'].lower())
    responses.append(intent['responses'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        
num_classes = len(labels)
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
vocab_size = 1000
embedding_dim =100
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

glove_path = "C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/glove.6B." + str(embedding_dim) + "d.txt"  # Change to the path of your GloVe file
embeddings_index = {}
with open(glove_path, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create an embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len,weights=[embedding_matrix], trainable=False))
model.add(SimpleRNN(16, activation='relu',kernel_regularizer=l2(0.01)))
#model.add(Dropout(0.5)) 
model.add(Dense(16, activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dense(16, activation='relu',kernel_regularizer=l2(0.01)))
#model.add(Dropout(0.1)) 
#model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

model.summary()

epochs = 500

history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)
# Plot training history
"""
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, training_labels, test_size=0.2, random_state=42)
test_loss, test_accuracy = model.evaluate(X_test, np.array(y_test))
print(f'Test Accuracy: {test_accuracy}')

model.save("chat_model")



# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

def check():
    
    confidence_threshold = 0.5
    print('start talking with bot, Enter quit to exit')
    while True:
        string = input('Enter: ')
        if string == 'quit': break
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([string]),
                                             truncating='post', maxlen=max_len))
        category = lbl_encoder.inverse_transform([np.argmax(result)]) # labels[np.argmax(result)]
        for i in data['intents']:
            if i['tag']==category:
                print(np.random.choice(i['responses']))
