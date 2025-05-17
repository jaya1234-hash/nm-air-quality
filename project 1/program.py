import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load the dataset
try:
    df = pd.read_csv('airquality-2.csv')  # Replace with actual filename
except FileNotFoundError:
    print("File not found. Make sure it's in the same directory or uploaded correctly.")
    sys.exit(1)

# Drop rows with missing values
df = df.dropna()

# Drop 'rownames' column if it exists
if 'rownames' in df.columns:
    df = df.drop(columns=['rownames'])

# Categorize Ozone values into AQI-like categories
def categorize_ozone(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    else:
        return 'Unhealthy'

df['AQI_Category'] = df['Ozone'].apply(categorize_ozone)

# Separate features and target variable
X = df.drop(['Ozone', 'AQI_Category'], axis=1)
y = df['AQI_Category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features for AQI Classification")
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('airquality-2.csv')

# Drop rows with missing values
df = df.dropna()

# Drop 'rownames' if it's just an index
if 'rownames' in df.columns:
    df = df.drop('rownames', axis=1)

# Define features and target
X = df.drop('Ozone', axis=1)
y = df['Ozone']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plotting predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Ozone")
plt.ylabel("Predicted Ozone")
plt.title("Actual vs Predicted Ozone Levels")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('airquality-2.csv')

# Drop rows with missing values
df = df.dropna()

# Drop 'rownames' column if present
if 'rownames' in df.columns:
    df = df.drop('rownames', axis=1)

# Use Temp as the feature (X) and Ozone as target (y)
X = df[['Temp']]
y = df['Ozone']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict values for the regression line
y_pred = model.predict(X)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Temperature (Temp)')
plt.ylabel
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('airquality-2.csv')

# Drop rows with missing Ozone values
df = df.dropna(subset=['Ozone'])

# Categorize Ozone levels
def categorize_ozone(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    else:
        return 'Unhealthy'

# Apply the categorization
df['Ozone_Category'] = df['Ozone'].apply(categorize_ozone)

# Count category occurrences
category_counts = df['Ozone_Category'].value_counts()

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['green', 'gold', 'red']
)
plt.title('Ozone Level Categories')
plt.axis('equal')  # Equal aspect ratio makes the pie chart circular
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Load the dataset
try:
    df = pd.read_csv('airquality-2.csv')  # Replace with actual filename
except FileNotFoundError:
    print("File not found. Make sure it's in the same directory or uploaded correctly.")
    sys.exit(1)

# Drop rows with missing values - essential for consistent data before plotting
df = df.dropna()

# Drop 'rownames' column if it exists and is not needed
if 'rownames' in df.columns:
    df = df.drop(columns=['rownames'])

# Univariate Analysis - Distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Ozone'], bins=30, kde=True, color='blue')
plt.title('Distribution of Ozone Levels')
plt.xlabel('Ozone')
plt.ylabel('Frequency')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Check if df is defined and contains required columns
if 'df' in globals() and all(col in df.columns for col in ['Month', 'Ozone']):
    # Drop rows with missing values in 'Month' or 'Ozone'
    df_clean = df.dropna(subset=['Month', 'Ozone'])

    # Ensure Month is treated as categorical for better plot labels
    df_clean['Month'] = df_clean['Month'].astype(str)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Month', y='Ozone', data=df_clean)
    plt.title('Ozone Levels Across Different Months')
    plt.xlabel('Month')
    plt.ylabel('Ozone')
    plt.show()
else:
    print("Error: DataFrame 'df' is not defined or missing required columns.")

import seaborn as sns
import matplotlib.pyplot as plt

# Multivariate Analysis - Correlation Matrix
plt.figure(figsize=(10, 8))

# Compute correlation matrix (only numeric columns are included by default)
corr_matrix = df.corr(numeric_only=True)

# Create heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Properly formatted title
plt.title('Correlation Matrix of Air Quality Features')

plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Multivariate Analysis - Pairplot
selected_features = ['Ozone', 'Solar.R', 'Wind', 'Temp']
sns.pairplot(df[selected_features], diag_kind='kde')

# Add a suptitle slightly above the plot
plt.suptitle('Pairplot of Selected Features', y=1.02)

plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example actual and predicted labels
y_true = [0, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("airquality-2.csv").dropna()

# Assume 'Ozone' is the target
X = df.drop(columns=['Ozone'])
y = df['Ozone']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "R2": r2}

# Create plot
metrics_df = pd.DataFrame(results).T
metrics_df[["MSE", "R2"]].plot(kind='bar', figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_performance.png")
plt.show()
