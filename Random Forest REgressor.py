from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import panadas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('file_name.csv')

# Remove rows with null values from the original DataFrame
df.dropna(inplace=True)

# removing outlier data
Q1 = df['column_name'].quantile(0.25)
Q3 = df['column_name'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['column_name'] >= lower_bound) &
        (df['column_name'] <= upper_bound)]


# finding min and max values
for i in df.columns:
    print(i, 'Min value :', df[i].min(), 'Max value :', df[i].max())

# Example data preparation code
X = df.drop('price', axis=1)  # Feature matrix
y = df['price']    # Target variable

# spliiting your data into your test and training sets. Reserving 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=200, random_state=42)
# Train the model on the training data
rf_regressor.fit(X_train, y_train)


# predition model
y_pred = rf_regressor.predict(X_test)
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Compute permutation importances & select out important factors
result = permutation_importance(
    rf_regressor, X, y, n_repeats=30, random_state=42)
selected_features = result.importances_mean > 0
X_selected = X[:, selected_features]

# Retrain the model one the newly selected criteria
new_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
new_rf_regressor.fit(X_selected, y)


y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
