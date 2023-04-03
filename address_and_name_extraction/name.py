import spacy
from spacy import displacy
import re


def extract_name(image_description):
    tokenized_desc = image_description.split(" ")
    print(tokenized_desc)
    nlp = spacy.load("xx_sent_ud_sm")

    found = False
    for iter in range(len(tokenized_desc)-2):
        if found:
            break
        word = tokenized_desc[iter] + " " + tokenized_desc[iter+1] + " " + tokenized_desc[iter+2]
        doc = nlp(word)

        for ent in doc.ents:
            if ent.label_ == "PER":
                return word
                
