'''
Name : Omkar Patil
Batch ID : DS/DA-16/06/2025
Topic : Hierarchical Clustering
'''
'''
CRISP-ML(Q) Process Model describes 6 phases :

1. Business and Data understanding
2. Data Prepration
3. Model Building and Tuining
4. Evaluation
5. Model Deployment
6. Monitoring and Maintainance.

# Business Problem:
    Airlines and airport authorities need to analyze passenger travel patterns, demand fluctuations,
    and terminal usage to improve operational efficiency, reduce congestion, and maximize revenue.

# High level Solution:
    By applying clustering techniques to passenger and operational data, airlines can segment travel patterns,
    identify peak demand groups, and optimize terminal resource allocation to enhance passenger satisfaction and profitability.

# Objective(s): Maximize the operational efficiency.
# Constraint(s): Maximize the financial health.

# Success Criteria:
    # Business Success Criteria: Increase the operational efficiency by 10% to 12% by segmenting the Airlines.
    # ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7.
    # Economic Success Criteria: The airline companies will see an increase in revenues by at least 8% (hypothetical numbers).

# Data Understanding:

    # Data Source : Data of Passengers Statistics is available with Airline companies and airport authorities.

# Data Dictionary:(Name of the feature -- Description -- Type -- Relevance)
1. Activity Period -- Period of activity represented in YYYYMM format -- Interval -- Helps track the timeframe of passenger traffic we can extract year and month for this feature
2. Operating Airline -- Name of the airline operating the flight -- Nominal -- relevant for clustering
3. Operating Airline IATA code -- 2 digit code for operaing airline -- Nominal -- not relevant as every airline as its unique IATA code
4. GEO Region -- Geographic region of the flight -- Nominal -- relevant
5. Terminal -- Airport terminal where the flight departs/arrives -- Nominal --relevant for clustering
6. Boarding Area -- Specific boarding gate area within the terminal -- Nominal -- relevant for clustering
7. Passenger Count -- Number of passengars recorded -- Ratio --relevant for clustering
8. Year -- Year extract from activity period -- Interval -- extract from activity period 
9. Month -- MOnth extract from activity.period -- Interval -- extract from activity period 

'''
# Code Modularity
# Installing Required libraries if not present
pip install sweetviz
pip install py-AutoClean
pip install clusteval
pip install sqlalchemy
pip install pymysql

# Importing required packages

import pandas as pd  # Importing Pandas library for data manipulation
import numpy as np   # Importing NumPy library for numerical computations
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting
import seaborn as sns  # Importing Seaborn library for plotting

import dtale      # Importing Dtale library for automated EDA (Exploratory Data Analysis)
import sweetviz  # Importing Sweetviz library for automated EDA (Exploratory Data Analysis)
from AutoClean import AutoClean  # Importing AutoClean library for automated data cleaning

from feature_engine.outliers import Winsorizer    # Importing Winsorizer for outlier treatment
from sklearn.preprocessing import OneHotEncoder    # Importing OneHotEncoder for encoding categorical(nominal) column

from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler for feature scaling
from sklearn.preprocessing import StandardScaler   # Importing StandardScalar for feature scaling
from sklearn.preprocessing import RobustScaler   # Importing RobustScalar for feature scaling
from sklearn.pipeline import make_pipeline  # Importing make_pipeline for creating a pipeline of preprocessing steps

from scipy.cluster.hierarchy import linkage, dendrogram  # Importing functions for hierarchical clustering
from sklearn.cluster import AgglomerativeClustering  # Importing AgglomerativeClustering for hierarchical clustering

from sklearn import metrics  # Importing metrics module from scikit-learn for evaluating clustering performance
from clusteval import clusteval  # Importing clusteval library for cluster evaluation

from sqlalchemy import create_engine, text  # Importing create_engine and text from sqlalchemy for database interaction
from urllib.parse import quote


# Reading .csv file into pandas DataFrame
passenger_stat = pd.read_csv(r"C:\Users\omkar\Downloads\Data Science Study Materials & Assignments\Data Science & ML\Data-set\Data Set  For Assignments\AirTraffic_Passenger_Statistics.csv")

# Credientials to connect MySQL Database
user = 'Omkar'     # user name
pw = quote('Killer@8080')    # password for user
db = 'DataScience_DB'    # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")     # Setup connection engine to database

# to_sql() - function to push dataframe to a sql table
passenger_stat.to_sql('passenger_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# to read the data from mysql database
sql = 'select * from passenger_tbl;'
df = pd.read_sql_query(text(sql), engine.connect())     # load data from mysql to df

# EDA - Exploratory Data Analysis
df.info()    # Data types
df.describe()    # EDA - Generating Discriptive Statistics (mean,median,std,count,etc)

sns.boxplot(df['Passenger Count'])     # plotting box plot to see weather outiers are present
plt.show()

s = sweetviz.analyze(df)    # Auto EDA using sweetviz library function
s.show_html("s.html")      # open the saved html to view interactive sweetviz dashboard


d = dtale.show(df, host = 'localhost', port = 8000)     # Auto EDA using Dtale
d.open_browser()     # open browser to view dtale dashboard 

# EDA Highlights
'''
- Missing Data : there are missing values in 'Operating Airline IATA Code'
- Duplicates : there are duplicate records
- Outliers : There are outliers in 'Passenger Count'
- Encoding : 'Operating Airlines', 'GEO Region', 'Terminal', 'Boarding Area' are categorical columns need to convert into numerical
- Type casting : 'Activity Period' should be converted to date time and 'Month' should be in numeric format
- Skewness & Kurtosis : 'Passenger Count' is positively skewed and is leptokurtic
'''

# Data preprocessing
'''
- Drop duplicate records.
- 'Activity Period' should be converted to date time and 'Month' to numeric format.
- Cleaning unwanted columns:
'Operating Airline IATA Code' is unique code assigned to operating airline and it has missing values we can drop it.
'Activity Period' also we can drop as we have year and month extracts from Activity Period.
- 'Passenger Count' outlier treatment has to be done.
- 'Operating Airline', 'GEO Region', 'Terminal', 'Boarding Area' one hot encoding has to be done.
- 'Year' convert to relative year.
- 'Month' treat with cyclic encoding using sine/cosine.
'''

df.drop_duplicates(inplace = True, ignore_index = True)  # dropping the duplicates and setting ignore_index to True to set the index to start from 0 again

# df['Activity Period'] = df['Activity Period'].astype(str)
df['Activity Period'] = pd.to_datetime(df['Activity Period'].astype(str), format = "%Y%m")    # Converting int64 to pandas datetime
df['Year'] = df['Activity Period'].dt.year     # Extracting year as type str from 'Activity Period'
df['Month'] = df['Activity Period'].dt.month      # Extracting month as type str from 'Activity Period'

df.drop(['Operating Airline IATA Code'], axis = 1, inplace = True)    # drop() to drop specific columns not required for modeling
df.drop(['Activity Period'], axis = 1, inplace = True)
df.info()    # Displaying non null values count and datatypes

winz_iqr = Winsorizer(capping_method = 'iqr',      # creating winsorizer model with capping method as 'iqr'
                      fold = 1.5,        # setting fold value to 1.5 to determine range for capping outliers based on IQR
                      tail = 'both',# both the tails of distriution will be capped
                      variables = ['Passenger Count']     # specifies the column in dataframe to apply winsorization 
                      )
df_winz = winz_iqr.fit_transform(df)    # fitting winsorizer model and transforming 'Passenger Count'

sns.boxplot(df_winz['Passenger Count'])     # plotting box plot to see weather all outiers are treated
plt.show()

# Creating an instance of the OneHotEncoder
enc = OneHotEncoder(sparse_output = False) # initializing method 
# setting sparse_output=False explicitly instructs the OneHotEncoder to return a dense array instead of a sparse matrix.

df = df[['Passenger Count', 'Year', 'Month', 'Operating Airline', 'GEO Region', 'Terminal', 'Boarding Area']]  # Rearranging the columns 
# Transforming the categorical columns (from Position column onwards) into one-hot encoded format and converting to DataFrame
df_enc = pd.DataFrame(enc.fit_transform(df.iloc[:, 3:]), columns = enc.get_feature_names_out(input_features = df.iloc[:, 3:].columns))
df_enc.columns = df_enc.columns.str.strip()     # removing leading and trailing spaces

# Treating 'Year' with relative year and 'Month' with cyclic encoding
df_winz['Relative Year'] = df_winz['Year'] - df_winz['Year'].min()     # converting 'Year' to 'Relative Year'
df_winz['Sine Month'] = np.sin(2 * np.pi * df_winz['Month'] / 12)    # encode 'Month' to 'Sine Month'
df_winz['Cosine Month'] = np.cos(2 * np.pi * df_winz['Month'] / 12)    # encode 'Month' to 'Cosine Month'
df_winz.drop(['Year', 'Month'], axis = 1, inplace = True)

df_clean = pd.concat([df_winz[['Passenger Count', 'Relative Year', 'Cosine Month', 'Sine Month']], df_enc], axis = 1)    # concating numerical columns with categorical columns after encoding
df_clean.info()     # Displaying concise summary of the cleaned DataFrame 'df_clean', including the number of non-null values and data types of each column
df_clean.head()     # Displaying first five records of updated cleaned dataframe 'df_clean'

pipe1 = make_pipeline(StandardScaler())       # Creating a pipeline using make_pipeline to apply StandardScaler for feature scaling
# Train the data preprocessing pipeline on data
# Applying the pipeline 'pipe1' to transform the cleaned DataFrame 'df_clean' and storing the transformed data in a new DataFrame 'df_pipelined'
df_pipelined = pd.DataFrame(pipe1.fit_transform(df_clean), columns = list(df_clean.columns), index = df_clean.index)

df_pipelined.head()     # Displaying first five records of transformed dataframe 'df_pipelined' to inspect changes
df_pipelined.describe()     # Generating discriptive statistics of transformed dataframe 'df_pipelined'
# validating feature scaling after scaling mean is 0 and std is 1

## END of Data preprocessing #

# Saving preprocessed data to sql database

df_pipelined.to_sql('preprocessed', con = engine, if_exists = 'replace', chunksize = 1000, index = False)   # pushing the cleaned dataframe 'df_pipelined' to sql database 'DataScience_DB' in table 'preprocessed'

# Model Building - Clustering Model
# Hierarchical Clustering - Agglomerative Clustering

plt.figure(figsize = (60, 8))  # creating a plot with figsize(60, 8)
tree_plot = dendrogram(linkage(df_pipelined, method = 'complete'))  # plotting dendrogram
plt.title('Hierarchical Clustering')  # setting title
plt.xlabel('Index')    # setting xlabel
plt.ylabel('Euclidean Distance')  # setting ylabel
plt.show()    # displaying the dendrogram plot

hc1 = AgglomerativeClustering(n_clusters = 6, metric = 'euclidean', linkage = 'single')  # creating a instance of agglomerative clustering with n_clusters set to 5 metric a euclidean and linkage single
y_hc1 = hc1.fit_predict(df_pipelined)    # fitting the model to dataframe 'df_pipelined'
cluster_labels = pd.Series(hc1.labels_)  # converting agglomerative clustering model generated labels to pandas series and storing it in cluster_label

metrics.silhouette_score(df_pipelined, cluster_labels)    # 0.8626 Silhouette score meets our ML success criteria for 6 clusters.

df_clust = pd.concat([cluster_labels, df], axis = 1)   # concating cluster labels to dataframe 'df' and storing it in dataframe 'df_clust'
df_clust = df_clust.rename(columns = {0: 'Cluster Label'})  # renaming the column with appropriate name

df_clust.to_sql('final', con = engine, if_exists = 'replace', chunksize = 1000, index = False)  # pushing the final dataframe 'df_clust' to sql database 'datascience_db' into 'final' table








