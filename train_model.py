import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('dataset.csv')

X = data[['length_url', 'https_token', 'ip', 'nb_at']]
y = data['status'].apply(lambda x: 0 if x == 'legitimate' else 1)

model = RandomForestClassifier()
model.fit(X.values, y)

pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved as model.pkl")