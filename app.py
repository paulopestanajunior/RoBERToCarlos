import pandas as pd
import streamlit as st
from streamlit_player import st_player
import re
import os
import random
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.models import load_model   # load saved model
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from transformers import BertTokenizer

import zipfile
import tempfile

stream = 'LSTM.zip'
if stream is not None:
  myzipfile = zipfile.ZipFile(stream)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    #st.info(f'trying to load model from tmp dir {model_dir}...')
    model = load_model(model_dir,  compile=False)


def load_tokenizer():
	return BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased') 

def normalize(data):
  """ Normalise (normalize) unicode data in Python to remove umlauts, accents etc. """
  return unicodedata.normalize('NFKD', data).encode('ASCII', 'ignore')

def clean_str(string):
    string  = normalize(string).decode('utf-8')
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')


    return string.strip().lower()


tokenizer = load_tokenizer()

st.image("berto.png")

felizes = ['https://www.youtube.com/watch?v=e187niM8fCo', #Quero Que Tudo v?? pro Inferno
           'https://www.youtube.com/watch?v=l0Dmgg6VZ80', #Lobo Mau
           'https://www.youtube.com/watch?v=FXBfoFrqc3s', #Eu sou Terr??vel
           'https://www.youtube.com/watch?v=fxtkVjBoDGg'  #Esse cara sou eu
          ]
tristes = ['https://www.youtube.com/watch?v=BrpIF_KCqhs', #N??o quero ver voc?? triste
           'https://www.youtube.com/watch?v=rzrH-bSSD2M', #Detalhes
           'https://www.youtube.com/watch?v=sS7dMnE30OM', #Emo????es
           'https://www.youtube.com/watch?v=tExaVNSjQdo'  #Amigo
          ]

# t??tulo
st.title("RoBERTo Carlos - Seu classificador de emo????es")

# subt??tulo
st.markdown("Est?? e uma aplica????o que utiliza deep learning para classificar sentimentos.")

sentence = st.text_input('Escreva aqui seu depoimento: ') 


# inserindo um bot??o na tela
btn_predict = st.button("Descubra se chorou ou se sorriu!")

# verifica se o bot??o foi acionado
if btn_predict or sentence:
  # Pre-process input
  sentence = clean_str(sentence)
  tokenize_words = tokenizer.tokenize(sentence)
  tokenize_words = tokenizer.encode(tokenize_words, return_tensors='pt')
  tokenize_words = pad_sequences(tokenize_words, maxlen=128, padding='post', truncating='post') 

  result = model.predict(tokenize_words)

  if result > 0.6:
    emocao = 'Positiva';
    musica = random.choice(felizes)
    texto = 'Que bom saber que est?? positivo hoje. Essa m??sica tem tudo a ver com seu momento!'
  else:
    emocao = 'Negativa';
    musica = random.choice(tristes)
    texto = 'Que pena saber que n??o est?? bem hoje. Essa m??sica pode te auxiliar nesse momento!'

  st.subheader("S??o tantas emo????es. E a sua ??: " +  emocao)

#  st.subheader(emocao)

  st.write(texto)

  st_player(musica)

