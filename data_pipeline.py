import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from preprocess import preprocess_text


def load_and_process_data():

    # Load dataset
    df = pd.read_csv(
        'dataset/spam.csv',
        encoding='latin-1'
    )

    # Keep useful columns
    df = df[['v1', 'v2']]

    # Rename columns
    df.columns = ['label', 'message']

    # Apply preprocessing
    df['final_text'] = df['message'].apply(
        preprocess_text
    )

    return df


def vectorize_data(df):

    # Create vectorizer
    vectorizer = CountVectorizer()

    # Convert text into vectors
    X = vectorizer.fit_transform(
        df['final_text']
    )

    return X, vectorizer