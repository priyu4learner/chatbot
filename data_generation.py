import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
import torch
import pandas

headers = { 'accept':'*/*',
'accept-encoding':'gzip, deflate, br',
'accept-language':'en-GB,en;q=0.9,en-US;q=0.8,hi;q=0.7,la;q=0.6',
'cache-control':'no-cache',
'dnt':'1',
'pragma':'no-cache',
'referer':'https',
'sec-fetch-mode':'no-cors',
'sec-fetch-site':'cross-site',
'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36',
 }

#To preprocess question and answer before entering into list 
def preprocess_text(str):
    str.lower()
    str=re.sub(r'\[[0-9]*\]',' ',str)
    str=re.sub(r'\s+',' ',str)
    return str

sen=[]      #List that contains questions and answers (questions followed by answers)
datas=''    #string that will contain the whole data in form of text for tokenization

#for tpo related queries
get_link=urllib.request.urlopen('https://tpo.mnnit.ac.in/tnp/company/faqs.php')
get_link=get_link.read()
dat=bs.BeautifulSoup(get_link,'lxml')
data=dat.find('div',id="accordion").find_all(class_='panel panel-default')
#datas+=dat.find('div',id="accordion").text

for d in data:
    question=d.find('h4').text
    answer=d.find('div',class_='panel-body').text
    sen.append(preprocess_text(question))
    sen.append(preprocess_text(answer))

#for faculty profile
get_link=urllib.request.urlopen('http://www.mnnit.ac.in/index.php/department/engineering/csed/csedfp')
get_link=get_link.read()
dat=bs.BeautifulSoup(get_link,'lxml')
data=dat.find('table').find_all('tr')
#datas=dat.find('table').text
c=0
for d in data:
    if c<74 and (c%8==0 or c%8==1 ):
        v=preprocess_text(d.find('td').text)
        sen.append(v)
    elif c>74 and (c%8==1 or c%8==2) :
        v=preprocess_text(d.find('td').text)
        sen.append(v)
    c=c+1

#about mnnit
get_link=urllib.request.urlopen('https://academics.mnnit.ac.in/new')
get_link=get_link.read()
dat=bs.BeautifulSoup(get_link,'lxml')
data=dat.find('div',class_="introduction").text
data=preprocess_text(data)
sens=nltk.sent_tokenize(data)
for s in sens:
    sen.append(s)
    sen.append(s)

#print(sen)

'''

data_text=''
for para in datas:
    data_text+=para
    data_text=data_text.lower()

data_text=data_text.lower()

data_text=re.sub(r'\[[0-9]*\]',' ',data_text)
data_text=re.sub(r'\s+',' ',data_text)

#print(data_text)
#sen=nltk.sent_tokenize(data_text)
#print(sen)
'''

'''
word=nltk.word_tokenize(data_text)
#print(word)
wnlem=nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return[wnlem.lemmatize(token) for token in tokens]

pr=dict((ord(punctuation),None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(pr)))
'''
torch.save(sen,'scrapdata.pth')