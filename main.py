import numpy as np
import statistics
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor

netflix_movies = pd.read_csv(r'C://Users//USER//Downloads//Documents//Best Movies Netflix.csv')

duplicateRows = netflix_movies[netflix_movies.duplicated()]

unique_df = netflix_movies.drop_duplicates()
# Generating random numbers for each row
netflix_movies['random'] = pd.Series(np.random.rand(len(netflix_movies)))

# Sorting the DataFrame by the random numbers
netflix_movies.sort_values(by='random', inplace=True)

# we will Choose the first 100 observations as your random sample
random_sample = netflix_movies.head(100)

# we will choose the columns with integer values for the mean
selected_columns = ['RELEASE_YEAR', 'SCORE','NUMBER_OF_VOTES']

# Creating a new DataFrame with only the selected columns
selected_data = random_sample[selected_columns]

# Now i can perform operations on this DataFrame
mean_values = selected_data.mean()

a = statistics.mean(mean_values)


# Printing the mean
print("Mean is :", a)
# let's Generate new random numbers
netflix_movies['random'] = pd.Series(np.random.rand(len(netflix_movies)))

# Sorting the DataFrame again
netflix_movies.sort_values(by='random', inplace=True)

# Choosing the next 100 observations
next_random_sample = netflix_movies.head(100)


# Creating a new DataFrame with only the selected columns
next_selected_data = next_random_sample[selected_columns]

# Now i can perform operations on this DataFrame
next_mean_values = next_selected_data.mean()


b = statistics.mean(next_mean_values)

# Printing the mean
print("next Mean is :", b)

# Plotting two features
plt.bar(random_sample['MAIN_GENRE'],random_sample['NUMBER_OF_VOTES'])
plt.xlabel('MAIN_GENRE')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('GENRE IMPACT ON NUMBER_OF_VOTES')
plt.show()
# Plotting two features
plt.bar(random_sample['MAIN_PRODUCTION'],random_sample['NUMBER_OF_VOTES'])
plt.xlabel('MAIN_PRODUCTION')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('MAIN_PRODUCTION IMPACT ON NUMBER_OF_VOTES')
plt.show()
# Plotting two features
plt.plot(random_sample['RELEASE_YEAR'],random_sample['NUMBER_OF_VOTES'])
plt.xlabel('RELEASE_YEAR')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('RELEASE_YEAR IMPACT ON NUMBER_OF_VOTES')
plt.show()
# Plotting two features
plt.bar(next_random_sample['MAIN_GENRE'],next_random_sample['NUMBER_OF_VOTES'])
plt.xlabel('MAIN_GENRE')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('GENRE IMPACT ON NUMBER_OF_VOTES')
plt.show()
# Plotting two features
plt.bar(next_random_sample['MAIN_PRODUCTION'],next_random_sample['NUMBER_OF_VOTES'])
plt.xlabel('MAIN_PRODUCTION')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('MAIN_PRODUCTION IMPACT ON NUMBER_OF_VOTES')
plt.show()
# Plotting two features
plt.plot(next_random_sample['RELEASE_YEAR'],next_random_sample['NUMBER_OF_VOTES'])
plt.xlabel('RELEASE_YEAR')
plt.ylabel('NUMBER_OF_VOTES')
plt.title('RELEASE_YEAR IMPACT ON NUMBER_OF_VOTES')
plt.show()


# Generate example data


# Performing one-sample t-test
t_statistic, p_value = stats.ttest_1samp(random_sample, popmean=25)
print("One-sample t-test: t-statistic = {t_statistic:.4f} , p-value = {p_value:.4f}")
# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(next_random_sample, popmean=25)
print("One-sample t-test: t-statistic = {t_statistic:.4f}, p-value = {p_value:.4f}")

# Generate example data


# Performing two-sample z-test
z_statistic, p_value = stats.ztest(random_sample, next_random_sample)
print("Two-sample z-test: z-statistic = {z_statistic:.4f}, p-value = {p_value:.4f}")

# Generate example contingency table

# Performing chi-square test of independence
chi2_statistic, p_value, _, _ = stats.chi2_contingency(random_sample)
print("Chi-square test: chi2-statistic = {chi2_statistic:.4f}, p-value = {p_value:.4f}")
# Performing chi-square test of independence
chi2_statistic, p_value, _, _ = stats.chi2_contingency(next_random_sample)
print("Chi-square test: chi2-statistic = {chi2_statistic:.4f}, p-value = {p_value:.4f}")

# Connecting to an SQLite database (creates a new file if it doesn't exist)
conn = sqlite3.connect('mydatabase.db')

# Creating a cursor object to execute SQL commands
cursor = conn.cursor()
# Writing DataFrame to SQL table
netflix_movies.to_sql('Best Movies Netflix', conn, if_exists='replace', index=False)



# let's look at the most and less popular movies in the united states,britain,india and japan
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='US' AND NUMBER_OF_VOTES=MAX(NUMBER_OF_VOTES) ;")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='US' AND NUMBER_OF_VOTES=AVG(NUMBER_OF_VOTES) ;")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='US' AND NUMBER_OF_VOTES=MIN(NUMBER_OF_VOTES);")

cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='GB' AND NUMBER_OF_VOTES=MAX(NUMBER_OF_VOTES) ;")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='GB' AND NUMBER_OF_VOTES=AVG(NUMBER_OF_VOTES);")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='GB' AND NUMBER_OF_VOTES=MIN(NUMBER_OF_VOTES);")

cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='IN' AND NUMBER_OF_VOTES=MAX(NUMBER_OF_VOTES);")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='IN' AND NUMBER_OF_VOTES=AVG(NUMBER_OF_VOTES);")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='IN' AND NUMBER_OF_VOTES=MIN(NUMBER_OF_VOTES);")

cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='JP' AND NUMBER_OF_VOTES=MAX(NUMBER_OF_VOTES);")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='JP' AND NUMBER_OF_VOTES=AVG(NUMBER_OF_VOTES);")
cursor.execute("SELECT MAIN_GENRE FROM Best Movies Netflix WHERE MAIN_PRODUCTION='JP' AND NUMBER_OF_VOTES=MIN(NUMBER_OF_VOTES);")


# Commiting the changes
conn.commit()

# Fetching all rows
rows = cursor.fetchall()
for row in rows:
    print(row)

# Closing the connection
conn.close()
#let's try to predict a upcoming netflix movie voting's voting based on its genre , production country and historical voting and scoring data 
X = pd.random_sample['MAIN_GENRE','MAIN_PRODUCTION']

y = pd.random_sample['NUMBER_OF_VOTES','SCORE','DURATION']
X_pred = pd.random_sample({
    'MAIN_GENRE': [thriller], 
    'MAIN_PRODUCTION': [us], 
})
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor
tree_regressor = DecisionTreeRegressor(max_depth=5)  # Adjust max_depth as needed



# Fit the model to the training data
tree_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

print(y_pred)
