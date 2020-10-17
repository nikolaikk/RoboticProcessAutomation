import re, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import joblib
from transformers.modeling_tf_distilbert import TFDistilBertModel
from transformers import DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
import tensorflow.keras.backend as K

tf.get_logger().setLevel('INFO')



def clean(text:pd.Series, remove_stop_words=True)->pd.Series:
    '''Clean text, removes irrelevant strings'''
    from nltk.corpus import stopwords

    text.fillna("fillna").str.lower()
    text = text.map(lambda x: re.sub("\[\[User.*",'',str(x)))
    text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    text = text.str.replace("?","").str.replace("'"," ").str.lower()
    # Unimportant part
    if remove_stop_words:
        text = text.map(lambda x: re.sub("(hi|hello|could|please|would|may)",'',str(x)))
        text = text.map(lambda x: re.sub(r"\b("+r"|".join(stopwords.words("english"))+r")\b",'',str(x)))
    return text


def get_tokenizer()->BertWordPieceTokenizer:
    '''Gets fast BERT tokenizer'''
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    save_path = 'distilbert_base_uncased/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tokenizer.save_pretrained(save_path)

    fast_tokenizer = BertWordPieceTokenizer('distilbert_base_uncased/vocab.txt', 
                                        lowercase=True)
    return fast_tokenizer

def tokenize(texts:pd.Series, tokenizer:BertWordPieceTokenizer, 
             chunk_size:int=240, maxlen:int=512)->np.array:    
    '''Tokenize input text, return in a form of array'''
    tokenizer.enable_truncation(max_length=maxlen)
    try:
        tokenizer.enable_padding(max_length=maxlen)
    except TypeError:
        tokenizer.enable_padding(length=maxlen)
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)


def build_dnn_model(transformer:TFDistilBertModel, max_len:int, num_classes:int=2,
                   lr=0.0007)->tf.keras.Model:
    '''Creates a keras model'''
    
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    
    embed = transformer.weights[0].numpy()
    embedding = tf.keras.layers.Embedding(np.shape(embed)[0], np.shape(embed)[1],
                          input_length=max_len, weights=[embed],
                          trainable=False)(input_word_ids)
    conc = K.sum(embedding, axis=2)
    conc = tf.keras.layers.Dense(128, activation='relu')(conc)
    conc = tf.keras.layers.Dense(256, activation='relu')(conc)
    conc = tf.keras.layers.Dense(128, activation='relu')(conc)
    
    conc = tf.keras.layers.Dense(num_classes, activation='softmax')(conc)
    loss = "categorical_crossentropy"
        
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=conc)
    
    model.compile(tf.keras.optimizers.Adam(lr=lr), 
                  loss=loss, 
                  metrics=['accuracy'])
    
    return model


def load_model_DNN1():
    return tf.keras.models.load_model("modelDNN1.h5")


def load_model_DNN2():
    return tf.keras.models.load_model("modelDNN2.h5")


def load_label_encoder():
    return joblib.load("labelencoder.pkl")