import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import plot_confusion_matrix

@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

feature_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe ):
	glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
	glass_type = glass_type[0]
	if glass_type == 1:
		return "building windows float processed".upper()
	elif glass_type == 2:
		return "building windows non float processed".upper()
	elif glass_type == 3:
		return "vehicle windows float processed".upper()
	elif glass_type == 4:
		return "vehicle windows non float processed".upper()
	elif glass_type == 5:
		return  "containers".upper()
	elif glass_type == 6:
		return "tableware".upper()
	else:
		return "headlamp".upper()

st.title("Glass Type predictiopn Web App.")
st.sidebar.title("Glass Type prediction Web App.")

if st.sidebar.checkbox("Show Raw Data"):
	st.subheader("Glass Type Data Set.")
	st.dataframe(glass_df)

st.sidebar.subheader("Visualisation selector")
plot_list = st.sidebar.multiselect("Select the Charts/Plots.", ('Correlation heatmap', 'Line chart', 'Area chart', 'Count plot', 'Pie chart','Box plot'))

if 'Line chart' in plot_list:
  # plot line chart
  st.subheader("Line Chart")
  st.line_chart(glass_df)

if 'Area chart' in plot_list:
  # plot area chart
  st.subheader("Area Chart") 
  st.area_chart(glass_df)
st.set_option('deprecation.showPyplotGlobalUse', False)

if 'Correlation heatmap' in plot_list:
  # plot correlation heatmap
  st.subheader("Correlation Heatmap")
  plt.figure(figsize = (12,6))
  sns.heatmap(glass_df.corr(), annot = True)
  st.pyplot()

if 'Count plot' in plot_list:
  # plot count plot
  st.subheader("Count Plot")
  plt.figure(figsize = (12,6))
  sns.countplot(x = "GlassType", data = glass_df)
  st.pyplot()

if 'Pie chart' in plot_list:
  # plot pie chart
  st.subheader("Pie Chart")
  plt.figure(figsize = (12,6))
  plt.pie(glass_df['GlassType'].value_counts(), labels = glass_df['GlassType'].value_counts().index, autopct = "%1.2f%%", startangle = 30, explode = np.linspace(.06, .12, 6))
  st.pyplot()

if 'Box plot' in plot_list:
  # plot box plot
  st.subheader("Box Plot")
  column = st.sidebar.selectbox("Select the column for box plot", ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'])
  plt.figure(figsize = (12,6))
  sns.boxplot(glass_df[column])
  st.pyplot()

st.sidebar.subheader("Select your values.")
ri = st.sidebar.slider("Input RI", float(glass_df['RI'].min()), float(glass_df['RI'].max()))
na = st.sidebar.slider("Input Na", float(glass_df['Na'].min()), float(glass_df['Na'].max()))
mg = st.sidebar.slider("Input Mg", float(glass_df['Mg'].min()), float(glass_df['Mg'].max()))
al = st.sidebar.slider("Input Al", float(glass_df['Al'].min()), float(glass_df['Al'].max()))
k  = st.sidebar.slider("Input K" , float(glass_df['K'].min()), float(glass_df['K'].max()))
ca = st.sidebar.slider("Input Ca", float(glass_df['Ca'].min()), float(glass_df['Ca'].max()))
ba = st.sidebar.slider("Input Ba", float(glass_df['Ba'].min()), float(glass_df['Ba'].max()))
fe = st.sidebar.slider("Input Fe", float(glass_df['Fe'].min()), float(glass_df['Fe'].max()))
si = st.sidebar.slider("Input Si", float(glass_df['Si'].min()), float(glass_df['Si'].max()))

st.sidebar.subheader("Choose the Classifier.")
classifier = st.sidebar.selectbox('classifier', ('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))



if classifier =='Support Vector Machine':
  st.sidebar.subheader("Model Hyperparameters.") 
  c_value = st.sidebar.number_input("C (Error Rate)", 1, 100, step = 1)
  kernel_input = st.sidebar.radio('Kernel', ('linear', 'rbf', 'poly'))
  gamma_input = st.sidebar.number_input('Gamma',1, 100, step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svm_model = SVC(C = c_value, kernel = kernel_input, gamma = gamma_input)
    svm_model.fit(X_train, y_train)
    score = svm_model.score(X_test, y_test)
    glass_type = prediction(svm_model, ri, na, mg, al, si, k, ca, ba, fe)
    
    st.write("Type of glass predicted is :", glass_type)
    st.write("Accuracy of the Model is:", round(score, 2))
    plot_confusion_matrix(svm_model, X_test, y_test)
    st.pyplot()

if classifier =='Random Forest Classifier':
  st.sidebar.subheader("Model Hyperparameters.") 
  n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input('Number of maximum depth of the Tree', 1, 100, step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Random Forest Classifier')
    rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
    rf_clf.fit(X_train, y_train)
    score = rf_clf.score(X_test, y_test)
    glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
    
    st.write("Type of glass predicted is :", glass_type)
    st.write("Accuracy of the Model is:", round(score, 2))
    plot_confusion_matrix(rf_clf, X_test, y_test)
    st.pyplot()

if classifier == 'Logistic Regression':
  st.sidebar.subheader("Model Hyperparameters")
  c_value = st.sidebar.number_input("C", 1, 100, step = 1 )
  max_iter_input = st.sidebar.number_input("Maximum Iterations", 10, 10000, 10)
  if sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    glass_type = prediction(lr, ri, na, mg, al, si, k, ca, ba, fe) 
    
    st.write("Type glass predicted is ", glass_type)
    st.write("Accuracy of the Model is:", round(score, 2))
    plot_confusion_matrix(lr, X_test, y_test)
    st.pyplot()
