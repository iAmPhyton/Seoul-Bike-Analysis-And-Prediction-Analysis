import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV 

#reading the dataset
seoulbike = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1') 

#defining features and target variable
x = seoulbike[['Temperature(°C)','Humidity(%)','Wind speed (m/s)','Hour','Holiday','Seasons']]
y = seoulbike['Rented Bike Count']

#spitting the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42) 

#initializing the model with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

#training the model
rf.fit(x_train, y_train)

#making predictions
y_pred = rf.predict(x_test)

#evaluating the model
#calculating Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R Squared Score", r2)

#understanding feature importance 
feature_importance = pd.Series(rf.feature_importances_, index=x.columns)
feature_importance.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importance in Random Forest Regressor')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

#HyperParameter Tuning
#defining the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10,15,20],
    'min_samples_split': [1,2,4]
}

#initialzing GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

#fitting the grid search to the data 
grid_search.fit(x_train, y_train)
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)