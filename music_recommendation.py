import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Sample Data
data = {
    'user_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'song_id': [101, 102, 103, 104, 105, 101, 102, 103, 104, 105],
    'timestamp': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'repeat_play': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Preprocess data
# Convert timestamps to datetime if necessary
# df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature Engineering: Create features such as the count of songs listened by the user
user_song_count = df.groupby('user_id')['song_id'].count().reset_index()
user_song_count.columns = ['user_id', 'song_count']
df = df.merge(user_song_count, on='user_id')

# Split data into features and target
X = df[['user_id', 'song_id', 'song_count']]
y = df['repeat_play']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Visualize feature importance
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()
