# Importamos las librerías necesarias
import random  # Para seleccionar respuestas aleatorias del chatbot
import json  # Para manejar archivos JSON donde se almacenan las intenciones del chatbot
import pickle  # Para cargar los archivos de palabras y clases previamente guardados
import numpy as np  # Para manejar operaciones matemáticas con arreglos

# Importamos Natural Language Toolkit (NLTK) para el procesamiento de texto
import nltk
from nltk.stem import WordNetLemmatizer  # Para reducir las palabras a su forma base (lematización)

# Importamos Keras para cargar el modelo de IA previamente entrenado
import keras
from keras.models import load_model  

# Inicializamos el lematizador
lemmatizer = WordNetLemmatizer()

# Cargamos los archivos generados en el código de entrenamiento
intents = json.loads(open('intents.json').read())  # Archivo JSON con las intenciones del chatbot
words = pickle.load(open('words.pkl', 'rb'))  # Lista de palabras procesadas en el entrenamiento
classes = pickle.load(open('classes.pkl', 'rb'))  # Lista de clases o intenciones posibles
model = load_model('chatbot_model.keras')  # Cargamos el modelo de red neuronal previamente entrenado

# Función para preprocesar una oración ingresada por el usuario
def clean_up_sentence(sentence):
    """
    Tokeniza la oración ingresada, lematiza cada palabra y la devuelve en una lista.
    """
    sentence_words = nltk.word_tokenize(sentence)  # Divide la oración en palabras
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lematiza cada palabra
    return sentence_words

# Función para convertir una oración en un "bag of words" (vector binario)
def bag_of_words(sentence):
    """
    Convierte una oración en un vector binario donde cada índice representa una palabra
    de la lista entrenada. Si la palabra está en la oración, el valor en el índice es 1, 
    de lo contrario, es 0.
    """
    sentence_words = clean_up_sentence(sentence)  # Procesamos la oración
    bag = [0] * len(words)  # Creamos un vector de ceros del tamaño del vocabulario entrenado
    for w in sentence_words:
        for i, word in enumerate(words):  
            if word == w:  
                bag[i] = 1  # Marcamos con 1 las palabras que aparecen en la oración
    print(bag)  # Mostramos el vector de palabras (solo para depuración)
    return np.array(bag)  # Devolvemos el vector como un array de numpy

# Función para predecir la intención de una oración
def predict_class(sentence):
    """
    Utiliza el modelo de red neuronal para predecir la intención de la oración ingresada.
    """
    bow = bag_of_words(sentence)  # Convertimos la oración en una maleta de palabras
    res = model.predict(np.array([bow]))[0]  # Hacemos la predicción con el modelo entrenado
    max_index = np.where(res == np.max(res))[0][0]  # Encontramos la categoría con mayor probabilidad
    category = classes[max_index]  # Obtenemos el nombre de la intención correspondiente
    return category  # Devolvemos la categoría de la intención detectada

# Función para obtener una respuesta del chatbot basada en la intención predicha
def get_response(tag, intents_json):
    """
    Busca en el archivo de intenciones la respuesta correspondiente a la intención detectada.
    """
    list_of_intents = intents_json['intents']  # Extraemos la lista de intenciones
    result = ""  
    for i in list_of_intents:
        if i["tag"] == tag:  # Buscamos la intención que coincide con la predicción
            result = random.choice(i['responses'])  # Seleccionamos una respuesta aleatoria
            break
    return result  # Devolvemos la respuesta seleccionada

# Función principal del chatbot para procesar una entrada y devolver la respuesta
def respuesta(message):
    """
    Recibe un mensaje del usuario, predice la intención y devuelve una respuesta del chatbot.
    """
    ints = predict_class(message)  # Predice la intención del mensaje
    res = get_response(ints, intents)  # Obtiene la respuesta adecuada
    return res  # Retorna la respuesta generada

# Bucle de interacción con el usuario
while True:
    message = input()  # Solicitamos al usuario que escriba un mensaje
    print(respuesta(message))  # Mostramos la respuesta generada por el chatbot
