import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

# load the model
model = load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

lemmatizer = WordNetLemmatizer()

def preprocess_user_input(user_input):

    # tokenize the user_input
    user_input_words = nltk.word_tokenize(user_input)

    # convert the user input into its root words : stemming
    stemmed_words = [lemmatizer.lemmatize(word) for word in user_input_words if word not in ignore_words]

    # Remove duplicacy and sort the user_input
    user_input_words = list(set(stemmed_words))
    user_input_words.sort()

    # Input data encoding : Create BOW for user_input
    vectorizer = CountVectorizer(vocabulary=words)
    user_input_bow = vectorizer.transform([user_input])

    return user_input_bow

def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label

def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
def bot_response(user_input):

    predicted_class_label =  bot_class_prediction(user_input)

    # extract the class from the predicted_class_label
    predicted_class = classes[predicted_class_label]

    # now we have the predicted tag, select a random response
    bot_responses = []
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            bot_responses.extend(intent['responses'])
    if bot_responses:
        # choose a random bot response
        bot_response = random.choice(bot_responses)
    else:
        bot_response = "I'm sorry, I don't have a response for that."

    return bot_response