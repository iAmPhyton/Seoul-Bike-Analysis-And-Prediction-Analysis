import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  

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

#run this section ONLY ONCE *AFTER* you have ran the section above this. it only works once. i don't know why

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