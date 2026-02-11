import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Example dataset of labeled text examples
# You can expand this with your own data
dataset = [
    "I love this product, it's amazing!",
    "This is terrible, I hate it.",
    "Pretty good, would recommend.",
    "Not bad, could be better.",
    "Absolutely fantastic experience!",
    "Worst purchase ever made.",
    "It's okay, nothing special.",
    "Great quality and fast delivery!",
    "Very disappointed with this.",
    "Excellent service and support!",
    "Poor quality, not worth it.",
    "Satisfied with my purchase.",
    "Completely useless product.",
    "Would buy again, very happy!",
    "Total waste of money.",
]

labels = [
    "positive",  # I love this product
    "negative",  # This is terrible
    "positive",  # Pretty good
    "neutral",   # Not bad
    "positive",  # Absolutely fantastic
    "negative",  # Worst purchase
    "neutral",   # It's okay
    "positive",  # Great quality
    "negative",  # Very disappointed
    "positive",  # Excellent service
    "negative",  # Poor quality
    "positive",  # Satisfied
    "negative",  # Completely useless
    "positive",  # Would buy again
    "negative",  # Total waste
]

# Create a TF-IDF vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Define a function to classify new text examples
def classify_text(text):
    """Classify text as positive, negative, or neutral
    
    Args:
        text (str): Text to classify
        
    Returns:
        str: Classification result (positive, negative, or neutral)
    """
    try:
        X_new = vectorizer.transform([text])
        prediction = clf.predict(X_new)
        return prediction[0]
    except Exception as e:
        print(f"Classification error: {e}")
        return "neutral"

# Test the classifier
if __name__ == "__main__":
    test_texts = [
        "This is wonderful!",
        "I'm not happy with this.",
        "It's fine, I guess."
    ]
    
    for text in test_texts:
        result = classify_text(text)
        print(f"Text: '{text}' -> Classification: {result}")
