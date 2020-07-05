#!/usr/bin/env python3
import fasttext
from pymystem3 import Mystem
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
import re
import os
import contextlib
fasttext.FastText.eprint = print



def preprocess_message(message, mystem, stopwords):
    #lemmatization
    tokens = mystem.lemmatize(message.lower())
    #remove stop words
    tokens = [token for token in tokens if token not in stopwords
              and token != " " 
              and token.strip() not in punctuation]
    
    result = " ".join(tokens)
    
    #remove numbers from the string
    result = re.sub(r'\d+', '', result)
    
    return result

if __name__== '__main__':

    print("Здравствуйте! Пожалуйста, напишите сообщение: ")
    
    #read message
    message = input()
    
    mystem = Mystem() 
    
    stopwords = stopwords.words("russian")
    
    #preprocess message
    processed_message = preprocess_message(message, mystem, stopwords)
    
    tokens = processed_message.split()
    
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model = fasttext.load_model('trained_model/model_20ep.sav')
    
    n_tokens = len(tokens)
    
    if  n_tokens == 0:
        print("Проверьте корректность сообщения")
    else:      
        average_vec = model.get_word_vector(tokens[0])
        for token in tokens[1:]:
            average_vec += model.get_word_vector(token)
            
        average_vec /= n_tokens
        avg_sum = sum(average_vec)
        
        if avg_sum == 0:
            print("Предсказанная категория: OTHER")
            print("Проверьте корректность сообщения, возможно вы сделали ошибку")
        else:
            pred = model.predict(processed_message, k=1)
    
            pred_label= pred[0][0].split('__label__')[1]
    
            print('Предсказанная категория: ', pred_label)