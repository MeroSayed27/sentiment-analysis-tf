from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

model = load_model("sentiment_model.h5")

with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

def encode_review(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in tokens]
    return pad_sequences([encoded], maxlen=500)

text = input("Enter a review: ")
encoded = encode_review(text)

prediction = model.predict(encoded)

if prediction[0][0] > 0.5:
    print("Positive")
else:
    print("Negative")