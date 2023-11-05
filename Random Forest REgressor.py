from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('data/houses_edited', index_col=None)
neighbourhood_to_code = dict(zip(df['city_district'], df['district_code']))




# Remove rows with null values from the original DataFrame
df.dropna(inplace=True)

# Converting data to ints that can be analyzed
columns_to_drop = ['final_price_transformed',
                   'final_price_log', 'full_link', 'full_address', 'title', 'mls', 'district_code', 'bedrooms']
df = df.drop(columns_to_drop, axis=1)
df = pd.get_dummies(df, columns=['city_district'])


# Loop to remove outliers in all columns
for column in df.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Prepare your data for modeling

X = df.drop('final_price', axis=1)
y = df['final_price']  # Target variable

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Compute permutation importances & select important features
result = permutation_importance(
    rf_regressor, X, y, n_repeats=30, random_state=42)
selected_features = result.importances_mean > 0
X_selected = X.iloc[:, selected_features]

# Retrain the model with the selected features
new_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
new_rf_regressor.fit(X_selected, y)

# Make predictions using the new model
y_pred = new_rf_regressor.predict(X_test)

# Calculate performance metrics for the new model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
d istrict_to_income = dict(zip(df['city_district'], df['mean_district_income']))