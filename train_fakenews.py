import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load both datasets
fake = pd.read_csv('archive/Fake.csv')
real = pd.read_csv('archive/True.csv')

# Add labels
fake['label'] = 1  # 1 = Fake
real['label'] = 0  # 0 = Real

# Combine them
df = pd.concat([fake, real], ignore_index=True)

# Use only the title (headline) column
df = df[['title', 'label']].dropna()

print("Dataset shape:", df.shape)
print("Class distribution:")
print(df['label'].value_counts())

X = df['title']
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFake News Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Save
joblib.dump(model, 'fakenews_model.pkl')
joblib.dump(vectorizer, 'fakenews_vectorizer.pkl')
print("\nFake News Model saved successfully!")
