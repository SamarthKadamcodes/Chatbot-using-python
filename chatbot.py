import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import keras
import tkinter as tk
from keras.models import load_model



lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
phq_score = 0
try:
    with open('user_info.json', 'r') as user_file:
        user_data = json.load(user_file)
except FileNotFoundError:
    user_data = {"user_name": None}
    user_data = {"user_age": None}


def save_user_data(user_data):
    with open('user_info.json', 'w') as user_file:
        json.dump(user_data, user_file)


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def collection_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    collection = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                collection[i] = 1
    return np.array(collection)


def predict_class(sentence):
    bow = collection_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_lists = []
    for r in results:
        return_lists.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_lists


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    result = None

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    if result is None:
        result = "I'm sorry, I didn't understand that."  # Provide a default response
    return result
class PHQ9Questionnaire:
    def __init__(self):
        self.questions = [
            "Little interest or pleasure in doing things?",
            "Feeling down, depressed, or hopeless?",
            "Trouble falling or staying asleep, or sleeping too much?",
            "Feeling tired or having little energy?",
            "Poor appetite or overeating?",
            "Feeling bad about yourself - or that you are a failure or have let yourself or your family down?",
            "Trouble concentrating on things, such as reading the newspaper or watching television?",
            "Moving or speaking so slowly that other people could have noticed? Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?",
            "Thoughts that you would be better off dead or of hurting yourself in some way?"
        ]
        self.answers = []

    def start_questionnaire(self):
        print("Please answer the following questions (0 for not at all, 1 for several days, 2 for more than half the days, 3 for nearly every day):")
        for question in self.questions:
            answer = input(f"{question} ")
            self.answers.append(int(answer))

    def get_answers(self):
        return self.answers


print("GO! bot is running!")

while True:
    message = input("You: ")
    user_name = user_data["user_name"]
    user_age = user_data["user_age"]
    phq_status = user_data["phq_nine_value"]
    if user_name is None:
        # If the user's name is not known, ask for it
        user_name = input("Chatbot: Hey there;)\nThis is the first time we are chatting.\nMay I know your name?\n")
        user_data["user_name"] = user_name
        save_user_data(user_data)
        print(
            f"Chatbot: Nice to meet you, {user_name}!\nI am a Chatbot, and my goal is to provide you with a safe and non-judgmental space\nwhere you can express your thoughts and feelings.\nYour well-being is my top priority, and I am here to listen,\nunderstand, and work together with you to address your concerns and challenges.")

    if user_age is None:
        user_age = input("Chatbot: How old are you? ")
        user_data["user_age"] = user_age
        save_user_data(user_data)
        print(f"Chatbot: Your age is {user_age}")

    if phq_status is None:
        phq9_questionnaire = PHQ9Questionnaire()
        phq9_questionnaire.start_questionnaire()
        phq9_answers = phq9_questionnaire.get_answers()
        phq_status = 1
        user_data["phq_nine_value"] = phq_status
        save_user_data(user_data)
        # Process PHQ-9 answers as needed in your chatbot code
        print("PHQ-9 Answers:", phq9_answers)
    ints = predict_class(message)
    res = get_response(ints, intents)
    res = res.replace("{name}", user_name)
    res = res.replace("{age}", user_age)

    print("Chatbot:", res)
