from pymongo import MongoClient
from html.parser import HTMLParser
import email
import string
import nltk
import os
import sys

# Establecer la conexión a MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client.dataset
dbEmails = db.emails
dbIndex = db.index

# reiniciar las tablas
dbEmails.delete_many({})
dbIndex.delete_many({})

def save_email_db(collection, id, text):
    document = {"idEmail": id, "text": text}
    result = collection.insert_one(document)    
    return True

def save_index_db(collection, label, email, id):
    document = {"label": label, "email": email, "id": id}
    result = collection.insert_one(document)    
    return True

DATASET_PATH = "/home/pablo/Downloads/MEISI/10_Inteligencia_Negocios/trabajo_final/posible/trec07p/data/"
INDEX_PATH = "/home/pablo/proyectos/scripts-python/src/in/data/full/index"

def process_indexs(n_elements):
    index = open(INDEX_PATH).readlines()
    for i in range(n_elements):
        mail = index[i].split(" ../")
        label = mail[0]
        path = mail[1].split("/")
        path = path[1][:-1]
        id = path.split(".")
        id = id[1]
        save_index_db(collection=dbIndex, label=label, email=path, id=id)
    return True

def process_emails():
  cursor = dbIndex.find()
  for document in cursor:
    emailPath = DATASET_PATH + document['email']

    with open(emailPath, 'r', encoding='utf-8', errors='ignore') as archivo:
      inmail = archivo.read()
    save_email_db(collection=dbEmails, id=document['email'], text=inmail)
    
   
process_indexs(75400)
process_emails()

print("GUARDADO OK")