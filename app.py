import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

df=pd.read_csv('heart_disease_data.csv')
df.head()

X=df.drop('target', axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model= LogisticRegression()
model.fit(X_train, y_train)
train_Prediction=model.predict(X_train)
Training_accuracy=accuracy_score(y_train, train_Prediction)

test_Prediction=model.predict(X_test)
Test_accuracy=accuracy_score(y_test, test_Prediction)

st.title("Heart Disease Prediction Model")
inputText= st.text_input('Provide comma seperated values')
sperated_input=inputText.split(',')
img= Image.open('img.jpg')
st.image(img, width=700)

try:
    np_df= np.asarray(sperated_input, dtype=float)
    reshaped=np_df.reshape(1,-1)
    prediction= model.predict(reshaped)

    if prediction[0]==0:
        st.markdown('***This Person did not have  any Heart Disease***')
    else:
        st.markdown('***This Person have Heart Disease***', unsafe_allow_html=True)
          
except ValueError:
  st.write('Please provide comma seperated values')

  st.subheader('About Data')
  st.write(df)

  st.subheader('Model Performance on Training Data')
  st.write(Training_accuracy)

st.subheader('Model Performance on Test Data')
st.write(Test_accuracy)


import matplotlib.pyplot as plt
st.subheader('Data Distribution')
plt.figure(figsize=(10, 6))

# Plot histogram for target
plt.hist(df['target'], bins=10, alpha=0.5, color='blue', edgecolor='black', label='Target')

# Plot histograms for all other columns in the same plot
for column in df.drop(columns='target').columns:
    plt.hist(df[column], bins=10, alpha=0.5, edgecolor='black', label=column)

plt.title('Histogram of Features')
plt.xlabel('y_test')
plt.ylabel('test_Prediction')
plt.legend()
st.pyplot(plt)

