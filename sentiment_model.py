# sentiment.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset (top 5000 words)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# Pad sequences so all reviews have same length
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)

# Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=32, input_length=500),
    LSTM(100),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model
model.save("sentiment_model.h5")
print("✅ Model trained and saved as sentiment_model.h5")


