import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

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
def prediction(model, f_columns):
	glass_type = model.predict([f_columns])
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
  