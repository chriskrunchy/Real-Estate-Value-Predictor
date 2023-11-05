

# import flask
# from flask import Flask, Response, jsonify, request
# #from flask_cors import CORS
# # from pydantic import BaseModel
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.inspection import permutation_importance
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset
# df = pd.read_csv('houses_edited.csv', index_col=0)

# # Data preprocessing code remains unchanged ...
# # Remove rows with null values from the original DataFrame
# df.dropna(inplace=True)

# # Converting data to ints that can be analyzed
# df = pd.get_dummies(df, columns=['type'])
# # df = pd.get_dummies(df, columns=['type'])

# # columns_to_drop = ['final_price_transformed',
#                     # 'final_price_log', 'full_link', 'full_address', 'title', 'mls', 'district_code', 'bedrooms', 'description']

# columns_to_drop = ['final_price_transformed',
#                     'final_price_log', 'full_link', 'full_address', 'title', 'mls', 'district_code', 'bedrooms', 'description', 'city_district']

# df = df.drop(columns_to_drop, axis=1)

# # filtering the middle 50% (wrt percentile)
# numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# for column in numeric_cols:
#     Q1 = df[column].quantile(0.25)
#     Q3 = df[column].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# # Prepare your data for modeling
# X = df.drop('final_price', axis=1)
# y = df['final_price']  # Target variable

# # Train-test split remains unchanged ...
# # Split your data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a Random Forest Regressor model
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(X_train, y_train)

# # Feature importance code remains unchanged ...
# # Compute permutation importances & select important features
# result = permutation_importance(
#     rf_regressor, X, y, n_repeats=30, random_state=50)
# selected_features = result.importances_mean > 0
# X_selected = X.iloc[:, selected_features]

# # # Retrain the model with selected features
# # new_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# # new_rf_regressor.fit(X_selected, y)

# # # Filter the test set to have only the selected features
# # X_test_selected = X_test[X_selected.columns]

# # # Use this to keep only the columns you need
# # features_columns = X_test_selected.columns

# # Retrain the model with the selected features
# new_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# new_rf_regressor.fit(X_selected, y)

# # Filter the test set to have only the selected features
# X_test_selected = X_test[X_selected.columns]

# # Make predictions using the new model
# #y_pred = new_rf_regressor.predict(X_test)
# y_pred = new_rf_regressor.predict(X_test_selected)

# # Use this to keep only the columns you need
# features_columns = X_test_selected.columns

# # app = FastAPI()

# app = Flask(__name__)
# # CORS(app)

# # # Define a Pydantic model to validate the input data
# # class HouseFeatures(BaseModel):
# #     list_price: float
# #     bathrooms: int
# #     sqft: float
# #     lat: float
# #     long: float
# #     mean_district_income: float
# #     bedrooms_ag: int
# #     bedrooms_bg: int

# # @app.route("/predict", methods=['GET', 'POST'])
# # def predict_final_price():
# #     # input_data = pd.DataFrame([[features.list_price, features.bathrooms, features.sqft, features.lat, features.long, features.mean_district_income, features.bedrooms_ag, features.bedrooms_bg]], columns=features_columns)
# #     features = requests.get_json();
    
# #     input_data = pd.DataFrame([
# #         [
# #             features.list_price,
# #             features.bathrooms,
# #             features.sqft,
# #             features.lat,
# #             features.long,
# #             features.mean_district_income,
# #             features.bedrooms_ag,
# #             features.bedrooms_bg
# #         ]
# #     ], columns=features_columns)
# #     predicted_price = new_rf_regressor.predict(input_data)
# #     return jsonify({"predicted_price": predicted_price[0]}), 200
# # @app.route("/predit", methods=["GET", "POST"])
# def predict_final_price(list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg):
#     # Construct a DataFrame from the input features

#     # json_temp = request.get_json();
#     # list_price = json_temp.get("list_price", 0)
#     # bathrooms = json_temp.get("bathrooms", 0)
#     # sqft = json_temp.get("sqft", 0)
#     # lat = json_temp.get("lat", 0)
#     # long = json_temp.get("long", 0)
#     # mean_district_income = json_temp.get("mean_district_income", 0)
#     # bedrooms_ag = json_temp.get("bedrooms_ag", 0)
#     # bedrooms_bg = json_temp.get("bedrooms_bg", 0)

#     input_data = pd.DataFrame([[list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg]],
#                               columns=features_columns)

#     # Use the trained model to predict
#     predicted_price = new_rf_regressor.predict(input_data)

#     # Return the predicted price
#     result = predicted_price[0]
#     return jsonify({"predicted_price": result}), 200

    
# # Example usage:
# predicted_price = predict_final_price(100000, 4, 2250, 43.905626, -79.449659, 70600, 2, 1)
# print(predicted_price)

# # print(predict_final_price(HouseFeatures(
# #     list_price=100000,
# #     bathrooms=4,
# #     sqft=2250,
# #     lat=43.905626,
# #     long=-79.449659,
# #     mean_district_income=70600,
# #     bedrooms_ag=2,
# #     bedrooms_bg=1)))

# if __name__ == "__main__":
#     app.run(debug=true)

# # NOTE: You don't need to run the server within the script when deploying. 
# # Instead, you will run the server from the command line.



# -*- coding: utf-8 -*-
"""random_forest_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R1egE_mGpumq8ojRHryjbbYrrDLJ_Zsj
"""

# df = pd.read_csv('/content/drive/My Drive/houses_edited.csv')



import flask
from flask import Flask, Response, jsonify, request
#from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
# CORS(app)
df = pd.read_csv('houses_edited.csv', index_col=0)

# Remove rows with null values from the original DataFrame
df.dropna(inplace=True)

# Converting data to ints that can be analyzed
# df = pd.get_dummies(df, columns=['type'])
# df = pd.get_dummies(df, columns=['type'])

# columns_to_drop = ['final_price_transformed',
                    # 'final_price_log', 'full_link', 'full_address', 'title', 'mls', 'district_code', 'bedrooms', 'description']

columns_to_drop = ['final_price_transformed',
                    'final_price_log', 'full_link', 'full_address', 'title', 'mls', 'district_code', 'bedrooms', 'description', 'city_district', 'type']

df = df.drop(columns_to_drop, axis=1)

# filtering the middle 50% (wrt percentile)
print("Setting up data...")
for column in df.columns:
    # Check type of first item in the column
    col_type = type(df[column].iloc[0])
    if col_type != np.bool_:
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

randomstate=50

print("Training data...")

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Compute permutation importances & select important features
result = permutation_importance(
    rf_regressor, X, y, n_repeats=30, random_state=randomstate)
selected_features = result.importances_mean > 0
X_selected = X.iloc[:, selected_features]

X_selected

# Retrain the model with the selected features
new_rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
new_rf_regressor.fit(X_selected, y)

# Filter the test set to have only the selected features
X_test_selected = X_test[X_selected.columns]

# Make predictions using the new model
#y_pred = new_rf_regressor.predict(X_test)
y_pred = new_rf_regressor.predict(X_test_selected)

# Calculate performance metrics for the new model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(mse**0.5)



# X_custom_input = pd.DataFrame(columns=X_test_selected.columns)
# # X_custom_input.loc[len(X_custom_input.index)] = [700000, 1, 612, 43.6701, -79.3862, 50000,1,1]
# X_custom_input.loc[len(X_custom_input.index)] = [1770750, 4, 2250, 43.905626, -79.449659, 70600 ,2,1]

# X_custom_input

# y_test_input = new_rf_regressor.predict(X_custom_input)

# y_test_input

# Use this to keep only the columns you need
features_columns = X_test_selected.columns
features_columns
print("Ready!")
# # Wrap the prediction logic inside a function
# def predict_final_price(list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg):
#     # Construct a DataFrame from the input features
#     input_data = pd.DataFrame([[list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg]],
#                               columns=features_columns)
#     print(input_data.head())
#     # Use the trained model to predict
#     predicted_price = new_rf_regressor.predict(input_data)

#     # Return the predicted price
#     return predicted_price[0]

# Define the columns explicitly
columns_for_prediction = ['list_price', 'bathrooms', 'sqft', 'lat', 'long', 'mean_district_income', 'bedrooms_ag', 'bedrooms_bg']

# def predict_final_price(list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg):
#     # Construct a DataFrame from the input features
#     input_data = pd.DataFrame([[list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg]],
#                               columns=columns_for_prediction)
#     print(input_data.head())
#     # Ensure we only use the columns present in the trained model
#     input_data = input_data[features_columns]
#     # Use the trained model to predict
#     predicted_price = new_rf_regressor.predict(input_data)

#     # Return the predicted price
#     return predicted_price[0]

# Example usage:
# predicted_price = predict_final_price(200, 4, 2250, 43.905626, -79.449659, 70600, 2, 1)
# print(predicted_price)

# print("hello")

@app.route("/predit", methods=["GET", "POST"])
def predict_final_price():
    # Construct a DataFrame from the input features

    json_temp = request.get_json();
    list_price = json_temp.get("list_price", 0)
    bathrooms = json_temp.get("bathrooms", 0)
    sqft = json_temp.get("sqft", 0)
    lat = json_temp.get("lat", 0)
    long = json_temp.get("long", 0)
    mean_district_income = json_temp.get("mean_district_income", 0)
    bedrooms_ag = json_temp.get("bedrooms_ag", 0)
    bedrooms_bg = json_temp.get("bedrooms_bg", 0)

    input_data = pd.DataFrame([[list_price, bathrooms, sqft, lat, long, mean_district_income, bedrooms_ag, bedrooms_bg]],
                              columns=features_columns)

    # Use the trained model to predict
    predicted_price = new_rf_regressor.predict(input_data)

    # Return the predicted price
    result = predicted_price[0]
    return jsonify({"predicted_price": result}), 200


if __name__ == "__main__":
    app.run(debug=true)