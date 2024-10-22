import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

#reading the dataset
seoulbike = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1') 
seoulbike 

#converting the date column to datetime format
seoulbike['Date'] = pd.to_datetime(seoulbike['Date'], format='%d/%m/%Y')

#extracting components from the date
seoulbike['Day'] = seoulbike['Date'].dt.day
seoulbike['Month'] = seoulbike['Date'].dt.month
seoulbike['Year'] = seoulbike['Date'].dt.year
seoulbike['Day of the Week'] = seoulbike['Date'].dt.dayofweek

#dropping the original date column
seoulbike = seoulbike.drop(columns=['Date'])

#re-encoding categorical columns and proceed with feature selection
seoulbike_encoded = pd.get_dummies(seoulbike, columns=['Holiday', 'Seasons'])
feature_cols = [col for col in seoulbike_encoded.columns if col != 'Rented Bike Count']
X = seoulbike_encoded[feature_cols]
y = seoulbike['Rented Bike Count']

#run this section ONLY ONCE *AFTER* you have ran the section above this. it only works once. i don't know why :)

#dropping the Functioning Day column as it is not needed (it caused some errors)
seoulbike = seoulbike.drop(columns=['Functioning Day'])

#applying one-hot encoding to both categorical columns
seoulbike_encoded = pd.get_dummies(seoulbike, columns=['Holiday', 'Seasons'])

#printing columns to see all available columns after encoding
print(seoulbike_encoded.columns)

#converting categorical columns to numerical using one-hot encoding
seoulbike_encoded = pd.get_dummies(seoulbike, columns=['Holiday', 'Seasons'], drop_first=True)
feature_cols = [col for col in seoulbike_encoded.columns if col != 'rentals']

#defining features and target
X = seoulbike_encoded[feature_cols]
y = seoulbike['Rented Bike Count'] 

#  GRID SEARCH

#train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing and fitting the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

#making predictions
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R Squared Score:", r2) 

#understanding feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importance in Random Forest Regreesor')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

#Hyperparameter Tuning
#defining the parameter grid

param_grid = {
    'n_estimators':[100, 200, 300],
    'max_depth': [10, 15 ,20],
    'min_samples_split': [2, 5, 20],
    'min_samples_leaf': [1, 2, 4]
}

#initializing Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

#fitting the grid search into the data
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_) 

# RANDOM FOREST REGRESSOR

#train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#initializing and fitting the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

#making predictions
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R Squared Score:", r2) 

#understanding feature importance
feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importance in Random Forest Regreesor')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show() 

#Hyperparameter Tuning
#defining the parameter grid

#initializing RandomizedSearchCV with 3-fold CV and parallel processing

param_grid = {
    'n_estimators':[50, 100, 150],
    'max_depth': [10, 15 ,20],
    'min_samples_split': [2, 5, 20],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions = param_grid, 
    n_iter=40,
    cv=3, 
    scoring='neg_mean_squared_error', 
    verbose=2, 
    n_jobs=-1, 
    random_state=42
)

#fitting the model
random_search.fit(X_train, y_train) 

# GRADIENT BOOSTING REGRESSOR

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#initializing the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42) 

#defining the paraneter grid for tuning 
param_grid = {
    'n_estimators': [50, 100, 150], 
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
} 

#performing Hyperparameter Tuning
#using RandomSearcgCV for faster tuning
random_search = RandomizedSearchCV(
    estimator=gbr,
    param_distributions = param_grid,
    n_iter = 30,
    cv = 3,
    scoring = 'neg_mean_squared_error',
    verbose = 2,
    n_jobs = -1,
    random_state = 42 
) 

#fitting the model
random_search.fit(X_train, y_train)

#getting best parameters and model
best_params = random_search.best_params_

print("Best Parameters:", best_params)

best_model = random_search.best_estimator_ 

#evaluating the model on the test set
#making predictions
y_pred = best_model.predict(X_test)

#calculating the evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) 

print("Mean Absolute Error:", mae)
print("R Squared Score:", r2) 

#MODEL COMPARISON AND ANALYSIS
#Comparing Perfomance Metrics (MAE, MSE, R Squared)
#relevant libraries have been imported at the beginning

best_rf_model = grid_search.best_estimator_
best_gb_model = random_search.best_estimator_

#Predictions for Random Forest
rf_predictions = best_rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

#Predictions for Gradient Boosting
gb_predictions = best_gb_model.predict(X_test)
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

print("RandomForestRegressor:")
print(f" - Mean Absolute Error: {rf_mae:.2f}")
print(f" - Mean Squared Error: {rf_mse:.2f}")
print(f" - R Squared Score: {rf_r2:.2f}\n")

print("Gradient Boosting Regressor:")
print(f" - Mean Absolute Error: {rf_mae:.2f}")
print(f" - Mean Squared Error: {rf_mse:.2f}")
print(f" - R Squared Score: {rf_r2:.2f}\n") 

#Plotting Predictions vs Actuals
#Plotting for Random Forest

plt.subplot(1, 2, 1)
plt.fill_between(np.arange(len(y_test)), y_test, rf_predictions, color='skyblue', alpha=0.5)
plt.plot(y_test, label='Actual', color='blue')
plt.plot(rf_predictions, label='Predicted', color='red')
plt.xlabel('Data Points')
plt.ylabel('Number of Bikes')
plt.title('Random Forest: Predictions vs Actuals')
plt.legend() 

#Plotting for Gradient Boosting

plt.subplot(1, 2, 2)

plt.fill_between(np.arange(len(y_test)), y_test, gb_predictions, color='lightgreen', alpha=0.5)
plt.plot(y_test, label='Actual', color='green')
plt.plot(gb_predictions, label='Predicted', color='orange')
plt.xlabel('Data Points')
plt.ylabel('Number of Bikes')
plt.title('Gradient Boosting: Predictions vs Actual')
plt.legend()

plt.tight_layout()
plt.show() 

#RESIDUAL ANALYSIS 
#Residuals for Random Forest
#Box plot for Random Forest Residuals
plt.figure(figsize=(12, 5))

#Calculating residuals (actual - predicted)
rf_residuals = y_test - rf_predictions
gb_residuals = y_test - gb_predictions 

# Residuals vs Predictions for Random Forest
plt.subplot(1, 2, 1)
plt.scatter(rf_predictions, rf_residuals, color='blue', alpha=0.5)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.title('Random Forest: Residuals vs Predictions')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals') 

