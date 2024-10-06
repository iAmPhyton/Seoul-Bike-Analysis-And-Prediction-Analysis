import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# read dataset 
seoulbike = pd.read_csv('SeoulBikeData.csv', encoding='ISO-8859-1') 

#creating a 'time_of_day' column based on the hour
def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'

#applying the function to categorize each hour
seoulbike['Time of the Day'] = seoulbike['Hour'].apply(get_time_of_day)

#calculating the average rentals grouped by season and time_of_day
avg_rentals = seoulbike.groupby(['Seasons', 'Time of the Day'])['Rented Bike Count'].mean().reset_index()

#creating plots to visualize the relationship between temperature and number of bikes rented 
#aggregating data to find the average Rented Bike Count for each temperature by Seasons
avg_rentals_temp = seoulbike.groupby(['Temperature(°C)', 'Seasons'])['Rented Bike Count'].mean().reset_index()

#creating line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_rentals_temp, x='Temperature(°C)', y='Rented Bike Count', hue='Seasons', palette='Set1')

#adding plot labels and title
plt.xlabel('Temperature (°C)')
plt.ylabel('Average Number of Bikes Rented')
plt.title('Average Number of Bikes Rented vs. Temperature by Season')

#displaying the plot
plt.legend(title='Seasons')
plt.show()

#creating visuals based on Humidity and the Number of Bikes by the season
plt.figure(figsize=(10, 6))
sns.lineplot(data=seoulbike, x='Humidity(%)', y='Rented Bike Count', hue='Seasons', palette='bright', linewidth=2) 
plt.xlabel('Humidity (%)')
plt.ylabel('Number of Bikes Rented')
plt.title('Relationship Between Humidity and Number of Bikes Rented by Season')
plt.legend(title='Seasons')

plt.show() 

#creating visuals to find the average Rented Bike Count for each wind speed by Seasons
avg_wind = seoulbike.groupby(['Wind speed (m/s)', 'Seasons'])['Rented Bike Count'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_wind, x='Wind speed (m/s)', y='Rented Bike Count', hue='Seasons', palette='bright', linewidth=2)
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Average Number of Bikes Rented')
plt.title('Average Number of Bikes Rented vs. Wind Speed by Seasons')
plt.legend(title='Seasons') 

plt.show() 

#creating visuals to display the Distribution of Rentals by Temperature and Holiday Status
plt.figure(figsize=(12, 6))
sns.lineplot(data=seoulbike, x='Temperature(°C)', y='Rented Bike Count', hue='Holiday', palette='husl', linewidth=1.5)
plt.xlabel('Temperature (°C)')
plt.ylabel('Number of Bikes Rented')
plt.title('Distribution of Rentals by Temperature and Holiday Status')
plt.legend(title='Is Holiday')
plt.show()

