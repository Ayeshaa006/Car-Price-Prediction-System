import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv('car_prediction_data.csv')

print("  _TOP 5 ROWS_  ")
print(df.head())
print()
print("  _BOTTOM 5 ROWS_  ")
print(df.tail())
print()
print("  _SHAPE_  ")
print("Rows:   ", df.shape[0])
print("Columns:", df.shape[1])
print()
print("  _NULL VALUES_  ")
print(df.isnull().sum())
print()
print("  _DATA TYPES_  ")
print(df.dtypes)
print()
print("  _STATISTICAL SUMMARY_  ")
print(df.describe())
print()
print("  _UNIQUE VALUES_  ")
print("Fuel Types:    ", df['Fuel_Type'].unique())
print("Seller Types:  ", df['Seller_Type'].unique())
print("Transmissions: ", df['Transmission'].unique())
print()


df['Car_Name']      = df['Car_Name'].fillna(df['Car_Name'].mode()[0])
df['Fuel_Type']     = df['Fuel_Type'].fillna(df['Fuel_Type'].mode()[0])
df['Seller_Type']   = df['Seller_Type'].fillna(df['Seller_Type'].mode()[0])
df['Transmission']  = df['Transmission'].fillna(df['Transmission'].mode()[0])
df['Selling_Price'] = df['Selling_Price'].fillna(df['Selling_Price'].mean())
df['Present_Price'] = df['Present_Price'].fillna(df['Present_Price'].mean())
df['Kms_Driven']    = df['Kms_Driven'].fillna(df['Kms_Driven'].mean())
df = df.dropna()

print("NULL VALUES AFTER FILLING:")
print(df.isnull().sum())
print()

df.drop('Car_Name', axis=1, inplace=True)

df['Year']       = df['Year'].astype(np.int64)
df['Owner']      = df['Owner'].astype(np.int64)
df['Kms_Driven'] = df['Kms_Driven'].astype(np.int64)

print("  _DATA TYPES AFTER CONVERSION_  ")
print(df.dtypes)
print()

df['Price_Category'] = pd.cut(
    df['Selling_Price'],
    bins=[0, 3, 8, 100],
    labels=['Low', 'Medium', 'High']
)
df.drop('Selling_Price', axis=1, inplace=True)

print("===== PRICE CATEGORY DISTRIBUTION =====")
print(df['Price_Category'].value_counts())
print()

x = df.iloc[:, 0:-1]
y = df['Price_Category']

print("X Shape:", x.shape)
print("Y Shape:", y.shape)
print()

cat_columns = x.select_dtypes(['object']).columns
x[cat_columns] = x[cat_columns].apply(lambda col: pd.factorize(col)[0])
y, labels = pd.factorize(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
print("Train Size:", X_train.shape)
print("Test Size: ", X_test.shape)
print()

# Random Forest
RF = RandomForestClassifier()
RF.fit(X_train, Y_train)
Y_rpred    = RF.predict(X_test)
r_accuracy = metrics.accuracy_score(Y_test, Y_rpred)
r_precision= metrics.precision_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
r_recall   = metrics.recall_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
r_f1       = metrics.f1_score(Y_test, Y_rpred, average='weighted', labels=np.unique(Y_rpred))
print("Random Forest  -> Accuracy: %.2f  Precision: %.2f  Recall: %.2f  F1: %.2f" % (r_accuracy, r_precision, r_recall, r_f1))

# Decision Tree
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train, Y_train)
Y_dpred    = Dtree.predict(X_test)
d_accuracy = metrics.accuracy_score(Y_test, Y_dpred)
d_precision= metrics.precision_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
d_recall   = metrics.recall_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
d_f1       = metrics.f1_score(Y_test, Y_dpred, average='weighted', labels=np.unique(Y_dpred))
print("Decision Tree  -> Accuracy: %.2f  Precision: %.2f  Recall: %.2f  F1: %.2f" % (d_accuracy, d_precision, d_recall, d_f1))

# Gaussian NB
GausNB = GaussianNB()
GausNB.fit(X_train, Y_train)
Y_gpred    = GausNB.predict(X_test)
g_accuracy = metrics.accuracy_score(Y_test, Y_gpred)
g_precision= metrics.precision_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
g_recall   = metrics.recall_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
g_f1       = metrics.f1_score(Y_test, Y_gpred, average='weighted', labels=np.unique(Y_gpred))
print("Gaussian NB    -> Accuracy: %.2f  Precision: %.2f  Recall: %.2f  F1: %.2f" % (g_accuracy, g_precision, g_recall, g_f1))

# KNN
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)
Y_kpred    = KNN.predict(X_test)
k_accuracy = metrics.accuracy_score(Y_test, Y_kpred)
k_precision= metrics.precision_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
k_recall   = metrics.recall_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
k_f1       = metrics.f1_score(Y_test, Y_kpred, average='weighted', labels=np.unique(Y_kpred))
print("KNN            -> Accuracy: %.2f  Precision: %.2f  Recall: %.2f  F1: %.2f" % (k_accuracy, k_precision, k_recall, k_f1))
print()

classifier_names = ['Random Forest', 'Decision Tree', 'Gaussian NB', 'KNN']

# Line Graph(all scores)
plt.figure(figsize=(12, 6))
plt.plot(classifier_names, [r_accuracy, d_accuracy, g_accuracy, k_accuracy],   marker='o', label='Accuracy')
plt.plot(classifier_names, [r_precision,d_precision,g_precision,k_precision],  marker='o', label='Precision')
plt.plot(classifier_names, [r_recall,   d_recall,   g_recall,   k_recall],     marker='o', label='Recall')
plt.plot(classifier_names, [r_f1,       d_f1,       g_f1,       k_f1],         marker='o', label='F1 Score')
plt.title("Scores of Applied Classifiers")
plt.xlabel("Classifiers")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig('static/line_graph.png')
plt.show()
print("Line graph saved.")

# Bar Graph(F1 Scores)
plt.figure(figsize=(10, 6))
plt.bar([1,2,3,4], [r_f1, d_f1, g_f1, k_f1],
        tick_label=classifier_names, width=0.6,
        color=['#08737f', '#089f8f', '#64c987', '#39b48e'])
plt.xlabel('Classifiers')
plt.ylabel('F1 Score')
plt.title('F1 Scores of Applied Classifiers')
plt.tight_layout()
plt.savefig('static/bar_graph.png')
plt.show()
print("Bar graph saved.")
print()

pickle.dump(RF, open('model.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))
print("Model saved as model.pkl")
print("Run app.py to start the Flask web application.")