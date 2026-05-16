from data_pipeline import (
    load_and_process_data,
    vectorize_data
)

# Load dataset
df = load_and_process_data()

# Vectorize
X, vectorizer = vectorize_data(df)

# Show vector shape
print("Vector Shape:")
print(X.shape)

# Show vocabulary preview
print("\nVocabulary:\n")
print(
    vectorizer.get_feature_names_out()[:20]
)