import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
import torch
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sen=torch.load('C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/scrapdata.pth')
data=torch.load('C:/Users/HP/OneDrive - MNNIT Allahabad, Prayagraj, India/Desktop/rnn_bot/chatbot/manual_data.pth')
for d in data:
    sen.append(d)

greeting_inputs=("hey","hello","good morning")
greeting_responses=["hey","hii","welcome"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
        
wnlem=nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return[wnlem.lemmatize(token) for token in tokens]

pr=dict((ord(punctuation),None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(pr)))

def generate_response(user_input):
    bot_response=''
    sen.append(user_input)

    word_vectorizer=TfidfVectorizer(tokenizer=get_processed_text,stop_words='english',use_idf=True)
    word_vectors=word_vectorizer.fit_transform(sen)
    similar_vector_values=cosine_similarity(word_vectors[-1],word_vectors)
    similar_sentence_number=similar_vector_values.argsort()[0][-2]
    
    matched_vector=similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched=matched_vector[-2]

    sen.pop()
    if vector_matched==0:
        bot_response=bot_response+"I am sorry I`m unable to understand"
        return bot_response
    else:
        if similar_sentence_number%2==0:
            bot_response=bot_response+sen[similar_sentence_number+1]
        else:
            bot_response=bot_response+sen[similar_sentence_number]
        return bot_response

'''
for testing purpose:
similar_vector_values=cosine_similarity(word_vectors[-1],word_vectors)
similar_sentence_number=similar_vector_values.argsort()[0][-2]
word_vectorizer=TfidfVectorizer(tokenizer=get_processed_text,stop_words='english',use_idf=True)
word_vectors=word_vectorizer.fit_transform(sen)

df_tfidf = pd.DataFrame(word_vectors.toarray())

print(df_tfidf)
'''
'''
continue_flag=True
print("Hello I`m MNNIT Bot, how can i help you")
while(continue_flag==True):
    human=input()
    human=human.lower()
    if human!='bye':
        if human=='thanks' or human=='thankyou':
            continue_flag=False
            print("Most welcome")
        else :
            if generate_greeting_response(human)!=None:
                print("MNNIT bot: ",generate_greeting_response(human))
            else:
                print("MNNIT bot: ",end="")
                print(generate_response(human))
                sen.remove(human) 
    else:
        continue_flag=False
        print("MNNIT bot says gooodbye")
'''
#print(len(sen))