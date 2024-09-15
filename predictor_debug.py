import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columns in the CSV file:", df.columns)
    return df

def preprocess_data(df):
    print("Initial shape:", df.shape)
    print("Missing values before drop:", df[['name', 'Food Group']].isna().sum())

    df = df.dropna(subset=["name", "Food Group"])
    print("Shape after dropping NaNs:", df.shape)

    # Inspect unique values in 'Food Group'
    print("Unique values in 'Food Group' before mapping:", df['Food Group'].unique())

    # Normalize and map food groups
    df['Food Group'] = df['Food Group'].str.strip().str.lower()
    label_mapping = {
        'baked foods': 0,
        'snacks': 1,
        'sweets': 2,
        'vegetables': 3,
        'american indian': 4,
        'restaurant foods': 5,
        'beverages': 6,
        'fats and oils': 7,
        'meats': 8,
        'dairy and egg products': 9,
        'baby foods': 10,
        'breakfast cereals': 11,
        'soups and sauces': 12,
        'beans and lentils': 13,
        'fish': 14,
        'fruits': 15,
        'grains and pasta': 16,
        'nuts and seeds': 17,
        'prepared meals': 18,
        'fast foods': 19,
        'spices and herbs': 20,
        'dairy and egg products ': 21
    }
    
    df['Food Group'] = df['Food Group'].map(label_mapping)
    print("Unique values in 'Food Group' after mapping:", df['Food Group'].unique())

    df = df.dropna(subset=['Food Group'])
    print("Shape after mapping and dropping NaNs:", df.shape)
    
    print("First few rows after preprocessing:", df.head())
    
    return df

if __name__ == "__main__":
    file_path = 'food_data.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    print("Food names after preprocessing:", df['name'].head())
