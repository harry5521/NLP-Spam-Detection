from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

from preprocess import preprocess_text

from data_pipeline import (
    load_and_process_data,
    vectorize_data
)

# Load dataset
df = load_and_process_data()

# Vectorize dataset
X, vectorizer = vectorize_data(df)

# Labels
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = MultinomialNB()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Model Accuracy:")
print(round(accuracy * 100, 2), "%")



# Save trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nModel and vectorizer saved successfully!")