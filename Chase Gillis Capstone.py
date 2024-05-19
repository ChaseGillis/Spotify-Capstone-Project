#Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu, normaltest, pearsonr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Set random seed (N14252303)
np.random.seed(14252303)

# Load data
data = pd.read_csv("Spotify 52k Data.csv")

# Data Preprocessing

# Handle missing values
data.dropna(inplace=True) 

# Convert 'explicit' from string to Boolean
data['explicit'] = data['explicit'].replace({'FALSE': False, 'TRUE': True})

# Handling Skewed Data
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'popularity'
skewed_features = data[features].skew().sort_values(ascending=False)
print("Skewness of features:\n", skewed_features)

# Handle skewed features, ensuring no negative values are passed to log1p
for feature in ['speechiness', 'liveness', 'instrumentalness']:
    # Make sure all values are positive before applying log1p
    if data[feature].min() <= 0:
        data[feature] += (-data[feature].min() + 1)  # Shift values to be positive
    data[feature] = np.log1p(data[feature])  # Now safe to apply log1p


# Question 1
# Normal Distributions?
# Histograms for checking normal distribution of features

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    sns.histplot(data[features[i]], kde=True, ax=ax)
    ax.set_title(features[i])
plt.tight_layout()
plt.show()

# Question 2
# Popularity vs Song Length
data['durationMins']=data['duration'] / 60000 # Convert from milliseconds to Minutes

# Plotting the data with outliers
plt.figure(figsize=(10, 6))
sns.regplot(x='durationMins', y='popularity', data=data, scatter_kws={'alpha':0.5})
plt.title('Relationship between Song Length and Popularity')
plt.ylim(0, 100)  # Set the limits for the y-axis
plt.xlim(data['durationMins'].min(), data['durationMins'].max()) 
plt.xlabel('Duration (Minutes)')
plt.ylabel('Popularity')
plt.show()

# Plotting the data without outliers
plt.figure(figsize=(10, 6))
sns.regplot(x='durationMins', y='popularity', data=data, scatter_kws={'alpha':0.5})
plt.title('Relationship between Song Length and Popularity')
plt.ylim(0, 100)  
plt.xlim(.5, 12)  
plt.xlabel('Duration (Minutes)')  
plt.ylabel('Popularity')
plt.show()

# Question 3
# T-test for explicit content effect on popularity
explicit_popular = data[data['explicit'] == True][target]
#print(explicit_popular)
non_explicit_popular = data[data['explicit'] == False][target]
t_stat, p_val = ttest_ind(explicit_popular, non_explicit_popular)
print(f'T-test result for explicit content on popularity: T-stat={t_stat}, P-value={p_val}')
# Calculate means
mean_popularity_explicit = data.groupby('explicit')['popularity'].mean().reset_index()

# Creating the bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='explicit', y='popularity', data=mean_popularity_explicit)
plt.title('Average Popularity by Explicit Content')
plt.xlabel('Explicit Content')
plt.ylabel('Average Popularity')
plt.xticks([0, 1], ['Non-Explicit', 'Explicit'])
plt.ylim(0,40)
plt.show()

# Question 4
# Check distribution of popularity for normality in both major and minor keys
major_popularity = data[data['mode'] == 1][target]
minor_popularity = data[data['mode'] == 0][target]

# Perform normality tests
major_normality = normaltest(major_popularity)
minor_normality = normaltest(minor_popularity)

print(f"Normality test for major key songs: Statistic={major_normality.statistic}, P-value={major_normality.pvalue}")
print(f"Normality test for minor key songs: Statistic={minor_normality.statistic}, P-value={minor_normality.pvalue}")

# Determine which test to use based on normality test results
if major_normality.pvalue > 0.05 and minor_normality.pvalue > 0.05:
    # Both groups are normally distributed, use T-test
    t_stat, p_val = ttest_ind(major_popularity, minor_popularity)
    print(f'T-test result for major vs minor key popularity: T-stat={t_stat}, P-value={p_val}')
else:
    # Use non-parametric Mann-Whitney U test if not normally distributed
    stat, p_val = mannwhitneyu(major_popularity, minor_popularity)
    print(f'Mann-Whitney U test result for major vs minor key popularity: Statistic={stat}, P-value={p_val}')

major_popularity = data[data['mode'] == 1][target]
minor_popularity = data[data['mode'] == 0][target]

# Calculate means
mean_popularity_mode = data.groupby('mode')['popularity'].mean().reset_index()

# Creating the bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x='mode', y='popularity', data=mean_popularity_mode)
plt.title('Average Popularity by Key Mode')
plt.xlabel('Key Mode')
plt.ylabel('Average Popularity')
plt.xticks([0, 1], ['Minor Key', 'Major Key'])
plt.ylim(25,35)
plt.show()

# Question 5
# Relationship Between Loudness and Energy
# Fill empty if there still are any for any reason
data['loudness'] = data['loudness'].fillna(data['loudness'].mean())


# scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='energy', y='loudness', data=data, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'})
plt.title('Energy vs. Loudness of Songs (with Regression Line)')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.show()


# Calculating Pearson correlation
correlation, p_value = pearsonr(data['energy'], data['loudness'])
print(f'Pearson correlation coefficient between energy and loudness: {correlation:.3f}')
print(f'P-value of the correlation: {p_value:.3e}')

# Interpretation of the result
if p_value < 0.05:
    print("The correlation between energy and loudness is statistically significant.")
else:
    print("There is no statistically significant correlation between energy and loudness.")

# Question 6
# Each Feature and Popularity
model_performance = {}
X = data[features]
y = data['popularity']

for feature in features:
    X_feature = X[[feature]]  # Isolate each feature
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=14252303)
    
    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_performance[feature] = r2_score(y_test, y_pred)  # Calculate and store R²

# Plotting R² values
plt.figure(figsize=(12, 6))
sns.barplot(x=list(model_performance.keys()), y=list(model_performance.values()))
plt.title('R² of Single Feature Models Predicting Popularity')
plt.xlabel('Features')
plt.ylabel('R² Value')
plt.xticks(rotation=45)
plt.show()

# Check outputs
print("R² Values for Each Feature:", model_performance)


# Question 7: Predicting popularity using all features
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14252303)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Question 7: R² for the model using all features: {r2:.4f}")

r2_instrumentalness = 0.023159955161881185  # compare the BEST R² from the regular features which was Instrumentalness

# Plotting comparison
plt.figure(figsize=(8, 4))
models = ['Instrumentalness', 'All Features Model']
r2_values = [r2_instrumentalness, r2]
sns.barplot(x=models, y=r2_values)
plt.title('Comparison of Model Performance')
plt.ylabel('R² Value')
plt.ylim(0, 0.3) 
plt.show()


# Question 8: Applying PCA and explaining variance
# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# PCA Dimension reduction
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Check if explained variance is non-zero
print("Explained variance:", explained_variance)

# Plot
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.title('PCA Explained Variance')
plt.show()


# Question 9: Logistic regression to predict major/minor key from valence
X_valence = data[['valence']].values.reshape(-1, 1)
y_key = data['mode']

# Fit model
model = LogisticRegression()
model.fit(X_valence, y_key)

# Generate data for the curve
x_test = np.linspace(data['valence'].min(), data['valence'].max(), 300)
y_prob = model.predict_proba(x_test.reshape(-1, 1))[:, 1]

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(data['valence'], y_key, alpha=0.2)
plt.plot(x_test, y_prob, color='red')  # Logistic regression curve
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Valence')
plt.ylabel('Probability of Major Key')
plt.show()

# Question 10: Using PCA components to predict whether a song is classical
# Encoding 'track_genre' as binary (1 if classical, 0 otherwise)
encoder = LabelEncoder()
data['is_classical'] = encoder.fit_transform(data['track_genre'] == 'classical')

# PCA
pca = PCA(n_components=5)  # Using 5 components since I found 5 meaningful ones in question 8
X_pca = pca.fit_transform(X_scaled)
y_classical = data['is_classical']

# Splitting the dataset for PCA model
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_classical, test_size=0.2, random_state=14252303)

# Logistic Regression using PCA components
log_reg_pca = LogisticRegression()
log_reg_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = log_reg_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)

# Preparing duration data
X_duration = data[['duration']]  # Using duration as a feature
X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X_duration, y_classical, test_size=0.2, random_state=14252303)

# Logistic Regression using duration
log_reg_dur = LogisticRegression()
log_reg_dur.fit(X_train_dur, y_train_dur)
y_pred_dur = log_reg_dur.predict(X_test_dur)
accuracy_dur = accuracy_score(y_test_dur, y_pred_dur)

print(f"Accuracy of predicting classical music using PCA components: {accuracy_pca:.4f}")
print(f"Accuracy of predicting classical music using duration: {accuracy_dur:.4f}")

plt.figure(figsize=(8, 4))
results = {'PCA Components': accuracy_pca, 'Duration': accuracy_dur}
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Comparison of Prediction Accuracies')
plt.ylabel('Accuracy')
plt.ylim(0.97,0.99)
plt.show()

#Extra Credit Clustering
# Applying K-means clustering
kmeans = KMeans(n_clusters=5, random_state=14252303)
clusters = kmeans.fit_predict(X_scaled)

# Applying PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, s=100, alpha=0.6, edgecolor='k')
plt.title('Clustering of Songs Based on Musical Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', loc='best')
plt.show()

# Adding the cluster labels to the original data
data['Cluster'] = clusters

# Calculate the mean of each feature for each cluster
cluster_means = data.groupby('Cluster')[features].mean()

# Plotting the mean features for each cluster
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Average Feature Values by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Features')
plt.show()
