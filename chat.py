import json 
import numpy as np
import nltk
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama 
colorama.init()
from colorama import Fore, Style, Back
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

with open("C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/intents.json") as file:
    data = json.load(file)
glove_path = "C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/glove.6B.100d.txt"  # Change to the path of your GloVe file
# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Remove stopwords and non-alphabetic characters
def preprocess_input(text, stop_words):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words]
    return ' '.join(filtered_words)

def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # load GloVe embeddings
    glove_embeddings = load_glove_embeddings(glove_path)

    # load stop words
    stop_words = set(stopwords.words('english'))

    # parameters
    max_len = 20
    confidence_threshold = 0.3
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

         # Preprocess user input
        preprocessed_input = preprocess_input(inp, stop_words)

        # Tokenize and pad user input
        user_sequence = tokenizer.texts_to_sequences([preprocessed_input])
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        confidence = np.max(result)
        print(confidence)
        if confidence >= confidence_threshold and any(i['tag'] == tag for i in data['intents']):
            for i in data['intents']:
                if i['tag'] == tag:
                    print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
        else:
            # Provide a default response when confidence is below threshold or category is not in the dataset
            print("I'm not sure how to respond. Can you please rephrase?")
        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
def test_run(text):
    model = keras.models.load_model('chat_model')
    confidence_threshold = 0.8
    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([text]),
                                             truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]
    confidence = np.max(result)
    
    if confidence >= confidence_threshold and any(i['tag'] == category for i in data['intents']):
        for i in data['intents']:
            if i['tag'] == tag:
                return np.random.choice(i['responses'])
    else:
        # Provide a default response when confidence is below threshold or category is not in the dataset
        print("I'm not sure how to respond. Can you please rephrase?")