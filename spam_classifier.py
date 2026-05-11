# =============================================
# Spam Email Classifier
# By: Mazhar Ali
# Description: Detects if an email is Spam
#              or Not Spam using Naive Bayes
# =============================================

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------
# STEP 1: Our Data (emails + labels)
# -----------------------------------------------
emails = [
    "Win a free iPhone now click here",
    "Congratulations you won a lottery",
    "Click here to claim your prize",
    "Free money waiting for you",
    "You have been selected for a reward",
    "Buy cheap medicines online now",
    "Make money fast from home",
    "Hot singles in your area click now",
    "Hey, are we still meeting tomorrow?",
    "Please find the attached report",
    "Can you review my code when free?",
    "Team meeting at 3pm today",
    "Your order has been shipped",
    "Let me know if you need help",
    "Happy birthday! Hope you have a great day",
    "The project deadline is next Monday",
    "Can we reschedule our call?",
    "I will send you the notes later",
    "Free gift cards available limited time",
    "Urgent your account needs verification",
]

labels = [
    "spam", "spam", "spam", "spam", "spam",
    "spam", "spam", "spam",
    "not spam", "not spam", "not spam", "not spam",
    "not spam", "not spam", "not spam", "not spam",
    "not spam", "not spam",
    "spam", "spam",
]

# -----------------------------------------------
# STEP 2: Convert text to numbers
# -----------------------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
y = labels

# -----------------------------------------------
# STEP 3: Split into training and testing
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# STEP 4: Train the Model
# -----------------------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------------------------
# STEP 5: Check Accuracy
# -----------------------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.1f}%")

# -----------------------------------------------
# STEP 6: Test with your own email
# -----------------------------------------------
print("\n--- Test Your Own Email ---")
user_email = input("Enter an email message: ")

transformed = vectorizer.transform([user_email])
result = model.predict(transformed)

print(f"\nResult: This email is --> {result[0].upper()}")

if result[0] == "spam":
    print("Warning: This looks like a spam email!")
else:
    print("This looks like a normal email!")