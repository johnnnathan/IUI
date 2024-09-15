import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


label_mapping = {
    'Baked Foods': 0,
    'Snacks': 1,
    'Sweets': 2,
    'Vegetables': 3,
    'American Indian': 4,
    'Restaurant Foods': 5,
    'Beverages': 6,
    'Fats and Oils': 7,
    'Meats': 8,
    'Dairy and Egg Products': 9,
    'Baby Foods': 10,
    'Breakfast Cereals': 11,
    'Soups and Sauces': 12,
    'Beans and Lentils': 13,
    'Fish': 14,
    'Fruits': 15,
    'Grains and Pasta': 16,
    'Nuts and Seeds': 17,
    'Prepared Meals': 18,
    'Fast Foods': 19,
    'Spices and Herbs': 20,
}
# Step 1: Load data from the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the CSV file:", df.columns)  # Debug: Print column names
    return df

# Step 2: Preprocess data
def preprocess_data(df):
    # Drop rows with missing values in 'name' or 'Food Group'
    df = df.dropna(subset=["name", "Food Group"])
    
    # Convert 'Food Group' to numerical labels
    df['Food Group'] = df['Food Group'].map(label_mapping)
    
    # Drop rows where mapping resulted in NaN
    df = df.dropna(subset=['Food Group'])
    
    # Convert 'name' to lowercase
    df['name'] = df['name'].str.lower()
    
    # Debug prints
    print("Sample food names:", df['name'].head())  # Check some names
    print("Number of rows after preprocessing:", len(df))  # Check the number of rows
    
    return df

# Step 3: Extract features using TF-IDF
def extract_features(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['name'])
    return X, vectorizer

# Step 4: Train and evaluate the model
def train_and_predict(df):
    # Prepare features and labels
    X, vectorizer = extract_features(df)
    y = df['Food Group']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model, vectorizer

# Step 5: Predict the food group for new items
def predict_food_group(model, vectorizer, food_names):
    X_new = vectorizer.transform(food_names)
    predictions = model.predict(X_new)
    return predictions



# Step 5: Predict the top 3 food groups for new items
def predict_top_3_food_groups(model, vectorizer, food_names):
    X_new = vectorizer.transform(food_names)
    # Predict probabilities for each class
    probabilities = model.predict_proba(X_new)
    
    # Get top 3 categories for each item
    top_3_predictions = []
    for prob in probabilities:
        # Get indices of top 3 probabilities
        top_3_indices = np.argsort(prob)[-3:][::-1]
        # Map indices to food group labels
        top_3_labels = [list(label_mapping.keys())[i] for i in top_3_indices]
        top_3_predictions.append(top_3_labels)
    
    return top_3_predictions
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'food_data.csv'  # Replace with your file path
    df = load_data(file_path)
    df = preprocess_data(df)
    
    # Train the model and evaluate
    model, vectorizer = train_and_predict(df)
    
    # Example: Predict the food group for new items
    new_items = ["Apple", "Chicken Salad", "pork chips", "milk cake", "chocolate milk"]

    top_3_predictions = predict_top_3_food_groups(model, vectorizer, new_items)
    for item, predictions in zip(new_items, top_3_predictions):
        print(f"Top 3 predictions for '{item}': {predictions}")
