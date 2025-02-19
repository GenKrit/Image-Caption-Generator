import string
import numpy as np
import os
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

def all_img_caption(filename):
    file=load_doc(filename)
    captions = file.split('\n')
    descriptons = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[:-2] = [caption]
        else:
            descriptions[img[:-2]].append(caption)
        return descriptions
    
def cleaning_text(captions):
    table=str.maketrans(",", string.punctuation)
    for img, caps in caption.items():
        for i,img_caption in enumerate(caps):
            img_caption.replace("-","-")
            desc=img_caption.split()
            desc=[word.lower() for word in desc]
            desc=[word.translate(table) for word in desc]
            desc=[word for word in desc if (len(word))>1]
            dsec=[word for word in desc if(word.isalpha())]

            img_caption=' '.join (desc)
            caption[img][i]=img_caption
    return captions