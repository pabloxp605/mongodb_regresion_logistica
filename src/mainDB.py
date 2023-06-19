from Parser import Parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score, precision_recall_curve
import scikitplot as skplt

import os
import pandas as pd

from pymongo import MongoClient
import matplotlib.pyplot as plt

import numpy as np


# ------ Establecer la conexión a MongoDB ------
URL_MONGODB = "mongodb://localhost:27017"
client = MongoClient(URL_MONGODB)
db = client.dataset
dbEmails = db.emails
dbIndex = db.index

# ------ Funciones auxiliares para preprocesamiento del conjunto de datos ------
def parse_email(text):
    p = Parser()
    pmail = p.parseFromString(text)
    return pmail

def prep_dataset_mongodb(n_elements):
    X = []
    y = []
    indexes = dbIndex.find().limit(n_elements)
    cursorIndex = list(indexes)
    for regIdx in cursorIndex:
        emailtxt = regIdx['email']
        label = regIdx['label']
        resultado = dbEmails.find_one({'idEmail': emailtxt})
        mail= parse_email(resultado['text'])
        X.append(" ".join(mail['subject']) + " ".join(mail['body']))
        y.append(label)
    return X, y

# ------------------------------------------------------------------------------------------------
# Leemos únicamente un subconjunto de 100 correos electrónicos
DATA_TRAIN = 100
X_train, y_train = prep_dataset_mongodb(DATA_TRAIN)
print(X_train)

#-------------------------------------------------------------------------------------------------

#Aplicamos la vectorización a los datos

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)

print(X_train.toarray())
print("\nFeatures:", len(vectorizer.get_feature_names_out()))

pd.DataFrame(X_train.toarray(), columns=[vectorizer.get_feature_names_out()])
print(y_train)

###### Entrenamiento del algoritmo de regresión logística con el conjunto de datos preprocesado
clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf)

#4. Predicción
#Lectura de un conjunto de correos nuevos
# Leemos 150 correos de nuestro conjunto de datos y nos quedamos únicamente con los 50 últimos
# Estos 50 correos electrónicos no se han utilizado para entrenar el algoritmo
DATA_PREDICT = 10000

X, y = prep_dataset_mongodb(DATA_PREDICT+DATA_TRAIN)
X_test = X[DATA_TRAIN:]
y_test = y[DATA_TRAIN:]

y_test_binary = [0 if 'ham' in string else 1 for string in y_test]

##### Preprocesamiento de los correos con el vectorizador creado anteriormente
X_test = vectorizer.transform(X_test)

##### Predicción del tipo de correo
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
y_pred_log_prob = clf.predict_log_proba(X_test)


##### Evaluación de los resultados
print("Predicción:\n", y_pred)
print("Predicción:\n", y_pred_prob)
print("\nEtiquetas reales:\n", y_test)
print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

# Calcular la tasa de error
error_rate = 1 - accuracy_score(y_test, y_pred)
print('Tasa de error: {:.3f}'.format(error_rate))

# Crear un DataFrame con los datos
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Crear una tabla de contingencia
contingency_table = pd.crosstab(df['Actual'], df['Predicted'])

# Mostrar la tabla de contingencia
print(contingency_table)

# TABLA DE CONTINGENCIA CON 50 DATOS DE PREDICCION
# ERROR
# (3 + 0 ) / 50 = O.O6
# ACIERTO
# (2 + 45 ) / 50 = 0.94

# Calcular la curva ROC
y_pred_prob_for_spam = y_pred_prob[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob_for_spam)
roc_auc = auc(fpr, tpr)

# Trazar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

# Generar la curva ROC y calcular el AUC
# skplt.metrics.plot_roc(y_test_binary, y_pred_prob)
# plt.show()


# Encontrar el valor de corte óptimo utilizando la métrica accuracy_score
f1_scores = []
cutoffs = []
for cutoff in np.arange(0.1, 1.0, 0.01):
    y_pred_binary = (y_pred_prob_for_spam >= cutoff).astype(int)
    f1Score = f1_score(y_test_binary, y_pred_binary)
    f1_scores.append(f1Score)
    cutoffs.append(cutoff)

optimal_cutoff = cutoffs[np.argmax(f1_scores)]
print("Valor de corte óptimo:", optimal_cutoff)




y_pred_new =  ["spam" if (value > optimal_cutoff) else "ham" for value in y_pred_prob_for_spam]


# Crear un DataFrame con los datos
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_new})

# Crear una tabla de contingencia
contingency_table1 = pd.crosstab(df1['Actual'], df1['Predicted'])

# Mostrar la tabla de contingencia
print(contingency_table1)

#TASA ACIERTO
# (3+45) / 50 = 0.96
#TASA ERROR
# (0 + 2) / 50 = 0.04



# Calcular la exactitud para diferentes valores de corte
cutoffs = np.arange(0.3, 0.9, 0.01)
accuracies = []
accuracy_ant=0
optimal_cutoff_2=0

for cutoff in cutoffs:
    y_pred_binary = (y_pred_prob_for_spam >= cutoff).astype(int)
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    accuracies.append(accuracy)
    if(accuracy > accuracy_ant):
        optimal_cutoff_2 = cutoff
    accuracy_ant = accuracy

print("Valor de corte óptimo segun acc es :", optimal_cutoff_2)

# Dibujar la curva de exactitud en función del valor de corte
plt.plot(cutoffs, accuracies, marker='.')
plt.xlabel('Cutoff')
plt.ylabel('Accuracy')
plt.title('Curva de Exactitud en función del Valor de Corte')
plt.show()

y_pred_new2 =  ["spam" if (value > optimal_cutoff_2) else "ham" for value in y_pred_prob_for_spam]

# Crear un DataFrame con los datos
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_new2})

# Crear una tabla de contingencia
contingency_table2 = pd.crosstab(df2['Actual'], df2['Predicted'])

# Mostrar la tabla de contingencia
print(contingency_table2)

