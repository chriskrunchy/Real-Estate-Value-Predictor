from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import panadas as pd

df = pd.read_csv('file_name.csv')

# Remove rows with null values from the original DataFrame
df.dropna(inplace=True)

#finding min and max values 
for i in df.columns:  
    print(i,'Min value :', df[i].min(),'Max value :', df[i].max())

# Example data preparation code
X = df.drop('price', axis = 1)  # Feature matrix
y = df['price']    # Target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
