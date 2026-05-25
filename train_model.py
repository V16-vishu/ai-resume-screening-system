import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

import pickle

# Load dataset
df = pd.read_csv("dataset/training_data.csv")

# Features
X = df['Resume Text']

# Labels
y = df['Job Role']

# Split data
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model pipeline
model = Pipeline([

    ('tfidf', TfidfVectorizer()),

    ('classifier', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Save model
pickle.dump(
    model,
    open("model.pkl", "wb")
)

print("Model trained successfully")