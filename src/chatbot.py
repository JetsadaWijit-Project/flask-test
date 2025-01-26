import numpy as np
import tensorflow as tf
import pickle
import random
import json

# Load model and preprocessors
model = tf.keras.models.load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)
with open("data/intents.json") as file:
    intents = json.load(file)

# Define max length (match the training input shape)
MAX_LEN = model.input_shape[1]  # Get the input shape from the model

# Chatbot response function
def chatbot_response(text):
    # Tokenize and pad the input sequence
    input_sequence = tokenizer.texts_to_sequences([text])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=MAX_LEN)
    
    # Predict and get the highest probability class
    prediction = model.predict(input_sequence)
    predicted_label = np.argmax(prediction)
    tag = label_encoder.inverse_transform([predicted_label])[0]
    
    # Get response from the intents file
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
