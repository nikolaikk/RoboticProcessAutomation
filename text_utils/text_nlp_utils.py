import nltk
import en_core_web_sm
nlp = en_core_web_sm.load()

def pos_tags(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent