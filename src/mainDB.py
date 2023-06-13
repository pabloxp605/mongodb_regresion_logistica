from Parser import Parser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
import os
import pandas as pd

from pymongo import MongoClient
import matplotlib.pyplot as plt


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

X_train, y_train = prep_dataset_mongodb(10)
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
X, y = prep_dataset_mongodb(700)
X_test = X[100:]
y_test = y[100:]

y_test_binary = [0 if 'ham' in string else 1 for string in y_test]

##### Preprocesamiento de los correos con el vectorizador creado anteriormente
X_test = vectorizer.transform(X_test)

##### Predicción del tipo de correo
y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)


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
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob[:, 1])
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