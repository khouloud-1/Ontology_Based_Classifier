# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:27:27 2021

@author: Djelloul BOUCHIHA, Abdelghani BOUZIANE, Noureddine DOUMI and Mustafa JARRAR
@paper: Ontology based Feature Selection and Weighting for Text Classification
"""

threshold = 0.8 # thrshold value [0..1]
wordnet_sim = 0    # WordNet based similarity: 0, 1,  2,  3,  4, 5, 6.

##################################  Preprocessing    ###########################################

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

# Preprocessing: Removing stop words, punctuations and digits, and Lemmatization
def clean_text(text):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words())
    
    punctuations="?:!.,;"
    
    # Remove digits
    text = ''.join(i for i in text if not i.isdigit())
    
    # Tokenization removes automatically white spaces
    text_tokens = nltk.word_tokenize(text)
    
    output_text_tokens =list()
    
    for word in text_tokens:
        # Remove punctuations
        if word in punctuations:
            text_tokens.remove(word)
        # Remove stop words
        if word in stop_words:
            text_tokens.remove(word)

    # Lemmatization
    for word in text_tokens:
        output_text_tokens.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
  
    return output_text_tokens


##################################  Extracting terms from WordNet synsets representing a class and their hyponyms ###########################################
# Method that returns all terms related to a class (domain) as tokens list
def concepts_domain(domain):
    D = wn.synsets(domain)
    terms_list_as_tokens = list()
    for s in D:
        for t in s.lemma_names():
            terms_list_as_tokens.append(t)
        h_D = s.hyponyms()
        for h in h_D:
            for p in h.lemma_names():
                terms_list_as_tokens.append(p)

    # Remove duplicates from terms_list_as_tokens
    terms_list_as_tokens = list(dict.fromkeys(terms_list_as_tokens))
    
    return terms_list_as_tokens

# Extracting terms
business_list = concepts_domain('business')
entertainment_list = concepts_domain('entertainment')
politics_list = concepts_domain('politics')
sport_list = concepts_domain('sport')
technology_list = concepts_domain('technology')

Global_terms_list = list(set(business_list + entertainment_list + politics_list + sport_list + technology_list))
        

##################################  WordNet based similarty metrics    ###########################################

from nltk.corpus import wordnet_ic
ic = wordnet_ic.ic('ic-brown.dat')
#ic = wordnet_ic.ic('ic-semcor.dat')

def similarity(sim, t1, t2):
    if (sim == 0):
        if (t1 == t2): return 1
        else: return 0
    else:
        try:
            syn1 = wn.synsets(t1)[0]
            syn2 = wn.synsets(t2)[0]
            if (sim == 1): return wn.path_similarity(syn1, syn2)    
            elif (sim == 2): return wn.lch_similarity(syn1, syn2)
            elif (sim == 3): return wn.wup_similarity(syn1, syn2)
            elif (sim == 4): return wn.res_similarity(syn1, syn2)
            elif (sim == 5): return wn.jcn_similarity(syn1, syn2)
            elif (sim == 6): return wn.lin_similarity(syn1, syn2)
        except: return 0


##################################  Ontology based Features Selection Technique (OFST)    ###########################################

def OFST(corpus, Global_list):
    vectors = list()
    for i in corpus:
        vector = list()
        for j in Global_list:
            n = 0;
            for k in i:
                if (similarity(wordnet_sim,j,k)>=threshold):
                    #print(j,' -------- ',k)
                    n = n + 1
            vector.append(n)
        vectors.append(vector)
    return vectors
#############################################################################



##################################  Read the corpus    ###########################################

import csv
import time 

start = time.time()

texts = list()
Y = list()

with open('Used_Corpus/BBC_Datasets/BBC_dataset.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        line = clean_text(row['text'])
        
        texts.append(line)
        if (row['category'] == 'business'): Y.append(0)
        elif (row['category'] == 'entertainment'): Y.append(1)
        elif (row['category'] == 'politics'): Y.append(2)
        elif (row['category'] == 'sport'): Y.append(3)
        elif (row['category'] == 'tech'): Y.append(4)

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# A term has to appear at least 1 time(s) in the corpus to be considered
processed_corpus = [[token for token in text if frequency[token] > 0] for text in texts]

############################################## calling OFST ##################################################

V = OFST(processed_corpus, Global_terms_list )

import numpy as np
V = np.transpose(np.array(V))
Y =  np.array(Y)

VY = np.vstack([V, Y])

OFST_mat = np.transpose(VY[:, :])

############################### Spliting the corpus #####################################

from sklearn.model_selection import train_test_split

XYtrain, XYtest = train_test_split(OFST_mat, test_size=0.3, train_size=0.7, shuffle=True)

Xtrain = XYtrain[:,:XYtrain.shape[1]-1]
Ytrain = XYtrain[:,XYtrain.shape[1]-1:]

Xtest = XYtest[:,:XYtest.shape[1]-1]
Ytest = XYtest[:,XYtest.shape[1]-1:]


############################### SVC #####################################

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# For the SVC function, the kernel parameter specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ (Radial Basis Function) will be used
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(Xtrain, np.ravel(Ytrain))

#######################################

end = time.time()

############################### SVC classifier evaluation #####################################

from sklearn.metrics import f1_score
y_true = Ytest[:,0]
y_pred = clf.predict(Xtest)

fs = f1_score(y_true, y_pred , average='micro')

print('\nCorpus size (dataset size): '+ str(OFST_mat.shape[0]) + ' documents')
print('\nNumber of features: '+ str(OFST_mat.shape[1]-1) + ' terms in synsets')
print('\nTime for preprocessing, OFST features extraction and SVC training: ',(end-start),' sec')
print('\nFor OFST and SVC, f1-score =                 '+str(fs))

############################### Classification Report #####################################

from sklearn.metrics import classification_report
print("\nClassification Report : precision, recall, F1 score for each of the classes '0- business', '1- entertainment', '2- politics', '3- sport' and '4- technology'")
target_names = ['class 0 (business)', 'class 1 (entertainment)', 'class 2 (politics)', 'class 3 (sport)', 'class 4 (technology)']
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

############################### Text  to classify #####################################
x = input('Enter your new english text: ')

vec = list()
ct = clean_text(x)
for i in Global_terms_list:
    n = 0
    for j in ct:
        if (similarity(wordnet_sim,i,j)>=threshold):
            #print(i,' -------- ',j)
            n = n + 1
    vec.append(n)

vv = list()
vv.append(vec)
vvvv = np.array(vv)

dec = clf.predict(vvvv)

if dec == 0: print('\n0- business')
elif dec == 1: print('\n1- entertainment')
elif dec == 2: print('\n2- politics')
elif dec == 3: print('\n3- sport')
elif dec == 4: print('\n4- technology')
