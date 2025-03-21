# Importamos las librerías necesarias
import random # Para mezclar datos aleatoriamente
import json # Para manejar archivos JSON
import pickle # PAra guardar y cargar objetos en archivos
import numpy as np # Para trabajar con arreglos y cálculos matemáticos

#Importar Natural Language ToolKit para procesamiento de texto
import nltk
from nltk.stem  import WordNetLemmatizer

from tensorflow.keras.models import Sequential # Para crear un modelo secuencial
from tensorflow.keras.layers import Dense, Dropout # Capas de la red neuronal
from tensorflow.keras.optimizers import SGD #O ptimizador Stochastic Gradient Descent

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

# Se carga el archivo JSON con las intenciones del chatbot
intents = json.loads(open('intents.json').read())

# Descarga de datos necesarios para NLTK
nltk.download('punkt') # Tokenizador de palabras
nltk.download('wordnet') # Diccionario de palabras para lematización
nltk.download('omw-1.4') # Datos adicionales para lematización

# Inicializa las listas para almacenar la infotmación
words = [] # Lista de palabras únicas
classes = [] # Lista de categorías o intenciones
documents= [] # Lista de pares (frase, intención)
ignore_letters = ['?','¿','!','¡','.',','] # Caracteres que se van a ignorar

# Procesamiento de los datos del archivo intents.json
for intent in intents ['intents']: # Recorre cada intención
    for pattern in intent['patterns']: # Recorre cada frase 
        word_list = nltk.word_tokenize(pattern) # Separamos las palabras
        words.extend(word_list) # Se agrega las palabras a la lista de palabras
        documents.append((word_list, intent["tag"])) # Se guarda la relación frase-intención
        if intent["tag"] not in classes: # Si la intención no está en la lista, es agregada
            classes.append(intent["tag"])

# Se lematiza y limpia las palabras, eliminando los duplicados
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words)) # Se eliminan los duplicados y se ordenan alfabéticamente

# Se guardan las palabras y clases en archivos para se utilizados más adelante
pickle.dump(words, open('words.pkl','wb')) # Para guardar las palabras
pickle.dump(classes, open('classes.pkl','wb')) # Para guardar las clases

# Se preparan los datos para entrenar la red neuronal
training= [] # Lista para almacenar los datos de entrenamiento
output_empty = [0]*len(classes) # Vector de salida de ceros (para clasificación)

# Se convierten los datos en una "maleta de palabras"
for document in documents:
    bag = []  # Creación de un vector para representar la frase
    word_patterns = document[0]  # Extracción de las palabras de la frase
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] # Lematiza
     # Se llena el vector de la "maleta de palabras"
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) # El 1 si se encuentra la palabra en la frase, si no 0

    # Se crea la salida esperada es decir categoria/intención
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1 # Se pone un 1 en la posición de la intención correspondiente

    training.append([bag,output_row])  # Se agrega los datos al conjunto de entrenamiento

# Se mezclan los datos de entrenamiento para evitar sesgos
random.shuffle(training) 
# Se convierte la lista en un array de NumPy
training = np.array(training, dtype=object)
print(training)

# Se dividen los datos en entrada (x) y salida (y)
train_x = list(training[:,0]) # Entradas (Maleta de palabras)
train_y = list(training[:,1]) # Salidas (intenciones)

# Se crea la red neuronal con Keras 
model = Sequential() # Modelo secuencial (capa por capa)
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # Capa oculta con 128 neuronas y activación Relu
model.add(Dropout(0.5))  # Para evitar sobreajuste (50% de neuronas desactivadas en cada iteración)
model.add(Dense(64, activation='relu')) # Segunda capa oculta con 64 neuronas y ReLU
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # Capa de salida con activación Softmax para clasificación

# Configuración del optimizador Stochastic Gradient Descent
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov= True)

# Compilación del modelo con función de perdida para clasificación multiclase
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# Entrenamiento del modelo con 100 épocas y lotes de tamaño 5
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

#Guardar el modelo entrenado en eun archivo . keras
model.save("chatbot_model.keras", train_process)
