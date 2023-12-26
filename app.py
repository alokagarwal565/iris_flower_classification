# Importing requisite libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Page configuration
st.set_page_config(
     page_title='Iris Flower Classification',
     page_icon='🌷',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🌷Iris Flower Classification')

# Load dataset from a local file
df = pd.read_csv(r'D:\Academic\project\iris.csv')

# Input widgets
st.sidebar.subheader('Input features')
sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.8)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.1)
petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.8)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.2)

# Separate to X and y
X = df.drop('Species', axis=1)
y = df['Species']  # Using single brackets to keep it as a Series

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Apply model to make predictions
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                           columns=X.columns)  # Use the columns from the original dataset
y_pred = lr.predict(input_data)

# Print EDA
st.subheader('Brief EDA')
st.write('The data is grouped by the class and the variable mean is computed for each class.')
groupby_species_mean = df.groupby('Species').mean()
st.write(groupby_species_mean)
st.line_chart(groupby_species_mean.T)

# Print input features
st.subheader('Input features')
input_feature = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(input_feature)

# Print prediction output
st.subheader('Output')
st.metric('Predicted class (using Logistic Regression)', y_pred[0], '')

# show_predictions_button = st.button("Prediction Using Other ML Models")

# if show_predictions_button:
    
#     # Additional Models
#     models = {
#     'Random Forest': RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42),
#     'Logistic Regression': LogisticRegression(),
#     'Support Vector Machine': SVC(),
#     'k-Nearest Neighbors': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Naive Bayes': GaussianNB()
#     }

#     # Apply model to make predictions for the additional models
#     predictions = {}
#     for name, model in models.items():
#      model.fit(X_train, y_train)
#      input_data_named = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
#                                      columns=X.columns)  # Use the columns from the original dataset
#      predictions[name] = model.predict(input_data_named)
    
#     st.subheader('Prediction using other ML Model')
    
#     # Print predictions for each model in a table
#     predictions_table = pd.DataFrame(predictions)
#     st.table(predictions_table)

#     # Print prediction probabilities for each model
#     st.subheader('Prediction Probabilities for Each Model')
#     probabilities_df = pd.DataFrame(index=lr.classes_)

#     for name, model in models.items():
#      try:
#         prediction_proba = model.predict_proba([[sepal_length, sepal_width, petal_length, petal_width]])
#         probabilities_df[name] = prediction_proba[0]
#      except AttributeError as e:
#         st.write(f'{name} does not support predict_proba.')

#     st.table(probabilities_df)