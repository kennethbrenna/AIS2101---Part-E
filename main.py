import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# PART 1
data = pd.read_csv("Top_Highest_Openings.csv")

# Thresholds for total Gross
blockbuster_threshold = 250000000 
hit_threshold = 100000000          

# Classifying movies
data['Total_Gross_Class'] = 'Flop' 
data.loc[data['Total Gross'] > blockbuster_threshold, 'Total_Gross_Class'] = 'Blockbuster'
data.loc[(data['Total Gross'] > hit_threshold) & (data['Total Gross'] <= blockbuster_threshold), 'Total_Gross_Class'] = 'Hit'

#Classifying openings
data['Opening_Class'] = 'Low'
data.loc[(data['Total_Gross_Class'] == 'Blockbuster') & (data['Opening'] > 120000000), 'Opening_Class'] = 'High'  
data.loc[(data['Total_Gross_Class'] == 'Blockbuster') & (data['Opening'] <= 120000000) & (data['Opening'] > 70000000), 'Opening_Class'] = 'Medium'  

data.loc[(data['Total_Gross_Class'] == 'Hit') & (data['Opening'] > 80000000), 'Opening_Class'] = 'High' 
data.loc[(data['Total_Gross_Class'] == 'Hit') & (data['Opening'] <= 80000000) & (data['Opening'] > 40000000), 'Opening_Class'] = 'Medium' 

data.loc[(data['Total_Gross_Class'] == 'Flop') & (data['Opening'] > 40000000), 'Opening_Class'] = 'High'
data.loc[(data['Total_Gross_Class'] == 'Flop') & (data['Opening'] <= 40000000) & (data['Opening'] > 20000000), 'Opening_Class'] = 'Medium' 

# Counting entries for both classes
total_gross_counts = data['Total_Gross_Class'].value_counts()
opening_class_counts = data['Opening_Class'].value_counts()

# Encoding Distributor with one-hot encoding
distributor_dummies = pd.get_dummies(data['Distributor'], prefix='Dist')
data = pd.concat([data, distributor_dummies], axis=1)

print("Total Gross Class Distribution:\n", total_gross_counts)
print("\nOpening Class Distribution:\n", opening_class_counts)

# Print first rows to show correct implementation
print(data[['Release', 'Opening', 'Total Gross', '% of Total', 'Theaters', 'Average', 'Distributor', 'Opening_Class', 'Total_Gross_Class']].head())

#Plotting of Total Gross Classes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
data['Total_Gross_Class'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Total Gross Classes')
plt.xlabel('Class')
plt.ylabel('Number of Movies')

# Plotting of Opening Classes
plt.subplot(1, 2, 2)
data['Opening_Class'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Opening Classes')
plt.xlabel('Class')
plt.ylabel('Number of Movies')

plt.tight_layout()
plt.show() 

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Opening', y='Total Gross', hue='Total_Gross_Class', data=data)
plt.title('Opening vs. Total Gross by Total Gross Class')
plt.show()

# Calculate statistics for Total Gross
total_gross_stats = data['Total Gross'].describe()
print(total_gross_stats)

from scipy.stats import mode
total_gross_mode = mode(data['Total Gross'])[0]
print("Mode:", total_gross_mode)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=data['Total Gross'])
plt.title('Box Plot of Total Gross Revenues')

plt.subplot(1, 2, 2)
sns.histplot(data['Total Gross'], bins=30, kde=True)
plt.title('Histogram of Total Gross Revenues')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Opening', y='Total Gross', hue='Total_Gross_Class', style='Opening_Class', data=data, palette='Set1', s=100)
plt.title('Scatter Plot of Opening vs. Total Gross with Class Labels')
plt.xlabel('Opening Revenue (in millions)')
plt.ylabel('Total Gross Revenue (in millions)')
plt.show() 


# PART 2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

data = data.drop(['Release', 'Title', 'Distributor', 'Date'], axis=1, errors='ignore')

# Encode Total_Gross_Class
le = LabelEncoder()
data['Total_Gross_Class_encoded'] = le.fit_transform(data['Total_Gross_Class'])

# Prepare X and Y
X = data.drop(['Total_Gross_Class', 'Total_Gross_Class_encoded', 'Opening_Class'], axis=1)
y = data['Total_Gross_Class_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predictions on test set
log_reg_pred = log_reg.predict(X_test)

# Apply SMOTE to better class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train the logistic regression model with different values for C
C_values = [0.1, 10, 1000]
for c in C_values:
    log_reg = LogisticRegression(max_iter=1000, C=c, penalty='l1', solver='liblinear')
    log_reg.fit(X_train_sm, y_train_sm) 
    log_reg_pred = log_reg.predict(X_test)
    print(f"Logistic Regression with C={c} (Using SMOTE)")
    print("Accuracy:", accuracy_score(y_test, log_reg_pred))
    print(classification_report(y_test, log_reg_pred))


experiments = [
    {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 10}, 
    {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 20},   
    {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2}
]

# Run the experiments
for i, params in enumerate(experiments, 1):
    rf_clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                    max_depth=params['max_depth'],
                                    min_samples_split=params['min_samples_split'],
                                    random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    print(f"Random Forest Experiment {i}: n_estimators={params['n_estimators']}, max_depth={'None' if params['max_depth'] is None else params['max_depth']}, min_samples_split={params['min_samples_split']}")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    
    
# PART 3
    
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


# Normalize the data before clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
def hierarchical_clustering(data, method, metric, title):
    Z = linkage(data, method=method, metric=metric)
    fig, ax = plt.subplots(figsize=(15, 5))
    dendrogram(Z, ax=ax)
    
    max_d = 5
    clusters = fcluster(Z, max_d, criterion='distance')
    print(f"Number of clusters formed: {np.unique(clusters).size}")
    silhouette_avg = silhouette_score(data, clusters)
    print(f"Silhouette Score: {silhouette_avg:.3f}")

# Experiment 1: Ward linkage with Euclidean distance
hierarchical_clustering(X_scaled, 'ward', 'euclidean', 'Ward Linkage with Euclidean')

# Experiment 2: Single linkage with Euclidean distance
hierarchical_clustering(X_scaled, 'single', 'euclidean', 'Single Linkage with Euclidean')

# Experiment 3: Complete linkage with Manhattan distance
hierarchical_clustering(X_scaled, 'complete', 'cityblock', 'Complete Linkage with Manhattan')


# Standardize the data
scaler2 = StandardScaler()
X_scaled2 = scaler2.fit_transform(data.select_dtypes(include=[np.number]))

# Define the DBSCAN experiemnts with different hyper parameters
dbscan_params = [
    {'eps': 0.5, 'min_samples': 5},
    {'eps': 0.8, 'min_samples': 5},
    {'eps': 0.3, 'min_samples': 10}
]

# Perform the experiments
for params in dbscan_params:
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    clusters = dbscan.fit_predict(X_scaled2)
    silhouette_avg = silhouette_score(X_scaled2, clusters)
    
    # Output results
    unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    print(f"DBSCAN with eps={params['eps']}, min_samples={params['min_samples']}")
    print(f"Number of clusters: {unique_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Noise points: {list(clusters).count(-1)}\n")  





    
  