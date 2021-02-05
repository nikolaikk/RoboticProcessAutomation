import numpy as np
import pandas as pd
from wit import Wit
import requests, json, re
import itertools, pprint

from config import ACCESS_TOKEN, ACCESS_TOKEN2


URL = "https://api.wit.ai/"
DATA_FILE = "data_transformed/data_updated.json"

auth = {"Authorization":"Bearer "+ACCESS_TOKEN}

params = {"limit":"1000"}

client = Wit(access_token=ACCESS_TOKEN)

def read_data():
    '''Reads json data stored on disk, uses DATA_FILE config variable.
    '''
    f = open (DATA_FILE, "r")
    data = json.loads(f.read())
    return data


def get_intents():
    '''Get all intents existing in current app.
    '''
    sub_url = "intents"
    response = requests.get(URL+sub_url, params=params, headers=auth)
    intents = [item["name"] for item in response.json()]
    return intents


def create_intent(intent_name):
    '''Created and add an intent to the app.
    - intent_name - intent name
    '''
    sub_url = "intents"
    if intent_name not in get_intents():
        data = json.dumps({"name":intent_name})
        response = requests.post(URL+sub_url, headers=auth, data=data)
        print(response.json())

def delete_intent(intent_name):
    '''Delete an intent from app.
    - intent_name - intent name
    '''
    sub_url = "intents"
    if intent_name in get_intents():
        response = requests.delete(URL+sub_url+"/"+intent_name, headers=auth)
        print(response.json())


def get_entities():
    '''Get all entities existing in current app.
    '''
    sub_url = "entities"
    response = requests.get(URL+sub_url, headers=auth, params=params)
    entities = [i["name"] for i in response.json()]
    return entities


def create_entity(entity_name):
    '''Created and add an entity to the app. Role is added as well,
    name of role is the same as name of entity, if role starts with
    'wit$', it is stripped.
    - entity_name - entity name
    '''
    sub_url = "entities"
    if entity_name not in get_entities():
        data = json.dumps({"name":entity_name,
                           "roles":[entity_name.replace("wit$","")]})
        response = requests.post(URL+sub_url, headers=auth, data=data)
        print(response.json())


def delete_entity(entity_name):
    '''Delete enetity if exists in app.
    - entity_name - name of entity
    '''
    sub_url = "entities"
    if entity_name in get_entities():
        response = requests.delete(URL+sub_url+"/"+entity_name, headers=auth)
        print(response.json())


def get_utterances():
    '''Get all intents from app.'''
    sub_url = "utterances"
    response = requests.get(URL+sub_url, params=params, headers=auth)
    utterances = response.json()
    return utterances


def create_utterance(text, intent=None, entities=None, traits=None):
    '''Create and add utterance to app. In order to use this method
    intent and entites should already exist in app.
    - text - utterance text
    - intent - intent of utterance
    - entities - dictionary of entities
    - traits - traits, not used yet
    '''
    sub_url = "utterances"
    if intent in get_intents():
        entity_list = []
        for entity_key, entity_value in entities.items():
            if entity_key in get_entities():
                position = [(match.start(), match.end()) for match in
                            re.finditer(entity_value, text)]
                # print(position)
                for pos in position:
                    entity_list.append({"entity":entity_key+":"+entity_key.replace("wit$",""),
                                        "start":pos[0],
                                        "end":pos[1],
                                        "body":entity_value,
                                        "entities":[],
                                        })
    data = [{"text":text,
            "intent":intent,
            "entities":entity_list,
            "traits":[]}]

    response = requests.post(URL+sub_url, headers=auth, data=json.dumps(data))
    print(response.json())


def delete_utterance(text):
    '''Delete utterance form app.
    - text - utterance text
    '''
    sub_url = "utterances"
    if text in [utterance["text"] for utterance in get_utterances()]:
        data = json.dumps([{"text":text}])
        response = requests.delete(URL+sub_url, headers=auth, data=data)
        print(response.json())


def train_once(data_sample):
    '''Train on a single data sampple if not trained on.
    data_sample - a dictionary of a data sample
    '''
    for entity in data_sample["entities"]:
        create_entity(entity)

    create_intent(data_sample["intent"])

    create_utterance(data_sample["text"],
                     data_sample["intent"],
                     data_sample["entities"])
    print(f"=== Trained on {data_sample['id']} {data_sample['text']}")


def train(data):
    '''Train on a list of dictionaries of data samples.
    - data - list of dictionaries od data samples
    '''
    for data_sample in data:
        train_once(data_sample)


def delete_all():
    '''Find and delete all entites, intents and utterances.'''
    for entity in get_entities():
        delete_entity(entity)
    for intent in get_intents():
        delete_intent(intent)
    for utterance in get_utterances():
        delete_utterance(utterance)


def predict(text):
    '''Predict text intents and entites based on trained app.
    - text - input text for prediction
    '''
    resp = client.message(text)
    # pprint.pprint(resp)
    return resp


if __name__ == '__main__':

    data = read_data()
    # train(data)
    resp = predict("Like a post that has 'we are here' on Linkedin")
    pprint.pprint(resp)
