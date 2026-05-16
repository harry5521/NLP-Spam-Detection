import pandas as pd

# Load dataset
df = pd.read_csv('dataset/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]

df.columns = ['label', 'message']

print(df['label'].value_counts())