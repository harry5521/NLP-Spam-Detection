import pickle

from preprocess import preprocess_text

# Load saved model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load saved vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Custom message
message = "You won free cash prize"

# Preprocess message
processed_message = preprocess_text(message)

# Convert into vector
message_vector = vectorizer.transform(
    [processed_message]
)

# Predict
prediction = model.predict(message_vector)

print("Message:")
print(message)

print("\nPrediction:")
print(prediction[0])