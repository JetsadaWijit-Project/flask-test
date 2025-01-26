import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import pickle

# Load intents
with open("data/intents.json") as file:
    data = json.load(file)

# Preprocess data
patterns = []
tags = []  # This should match patterns in size
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])  # Add tag for each pattern
    responses[intent["tag"]] = intent["responses"]

# Encode labels (tags)
encoder = LabelEncoder()
tags_encoded = encoder.fit_transform(tags)  # This matches patterns in size

# Save the encoder for later use
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(patterns)
X = tf.keras.preprocessing.sequence.pad_sequences(X)

# Save tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Convert tags to numpy array
y = np.array(tags_encoded)

# Build model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(set(tags)), activation="softmax")  # Use tags here
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save("chatbot_model.h5")
