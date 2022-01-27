# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:24:30 2021

@author: Djelloul BOUCHIHA, Abdelghani BOUZIANE, Noureddine DOUMI and Mustafa JARRAR
@paper: Ontology based Feature Selection and Weighting for Text Classification
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:30:00 2021

@author: Djelloul BOUCHIHA, Abdelghani BOUZIANE, Noureddine DOUMI and Mustafa JARRAR
@paper: Ontology based Feature Selection and Weighting for Text Classification
"""

##################################  Preprocessing    ###########################################

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def clean_text1(text):
    
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


##################################  Read the corpus    ###########################################

import csv
import time 

start = time.time()

texts = list()
Y = list()

with open('Used_Corpus/BBC_Datasets/BBC_dataset.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        line = clean_text1(row['text'])
        
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



############################### Doc2Vec #####################################

import gensim 
import numpy as np

# Create the tagged document needed for Doc2Vec
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

train_data = list(create_tagged_document(processed_corpus))

#print(train_data[:])


# Init the Doc2Vec model /  vector_size corresponds to the number of features
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

D = list()
for i in processed_corpus:
    D.append(model.infer_vector(i))


y=np.array(D)
y = np.transpose(y[:, :])

D2V = np.vstack([y, Y])
D2V = np.transpose(D2V[:, :])


from sklearn.model_selection import train_test_split

XYtrain, XYtest = train_test_split(D2V, test_size=0.3, train_size=0.7, shuffle=True)


Xtrain = XYtrain[:,:XYtrain.shape[1]-1]
Ytrain = XYtrain[:,XYtrain.shape[1]-1:]
#print("\n****** Xtrain  ******")
#print(Xtrain)
#print("\n****** Ytrain  ******")
#print(Ytrain)


Xtest = XYtest[:,:XYtest.shape[1]-1]
Ytest = XYtest[:,XYtest.shape[1]-1:]
#print("\n****** Xtest  ******")
#print(Xtest)
#print("\n****** Ytest  ******")
#print(Ytest)


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


print('\nCorpus size (dataset size): '+ str(D2V.shape[0]) + ' documents')
print('\nNumber of features (vector size): '+ str(D2V.shape[1]-1))
print('\nTime for preprocessing, the training data for Doc2Vec (features extraction) and SVC training: ',(end-start),' sec')
print('\nFor Doc2Vec and SVC, f1-score =                 '+str(fs))

############################### Classification Report #####################################

from sklearn.metrics import classification_report
print("\nClassification Report : precision, recall, F1 score for each of the classes '0- business', '1- entertainment', '2- politics', '3- sport' and '4- technology'")
target_names = ['class 0 (business)', 'class 1 (entertainment)', 'class 2 (politics)', 'class 3 (sport)', 'class 4 (technology)']
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

############################### Text  to classify #####################################
x = input('Enter your new english text: ')

x_vec = [model.infer_vector(clean_text1(x))]


dec = clf.predict(x_vec)

if dec == 0: print('\n0- business')
elif dec == 1: print('\n1- entertainment')
elif dec == 2: print('\n2- politics')
elif dec == 3: print('\n3- sport')
elif dec == 4: print('\n4- technology')





