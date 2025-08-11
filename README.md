Survival-at-Sea-An-Exploratory-Data-Analysis-of-the-Titanic-Dataset
Survival at Sea: An Exploratory Data Analysis of the Titanic Dataset I have Done this task by using the tools Python (Pandas, Matplotlib, Seaborn)

🚢 Titanic Dataset - Exploratory Data Analysis (EDA)
This project performs an exploratory data analysis on the Titanic dataset using Python libraries such as Pandas, Matplotlib, and Seaborn. The goal is to uncover patterns and relationships that influenced passenger survival during the Titanic disaster.

📂 Repository Structure
titanic-eda/
├── titanic_eda.ipynb       # Jupyter Notebook with full analysis
├── titanic_eda.pdf         # PDF report of findings
├── README.md               # Project overview
└── requirements.txt        # Python dependencies


📊 Key Analysis Steps
- Data loading and preview
- Missing value inspection
- Age distribution visualization
- Survival analysis by gender and class
- Correlation matrix of numerical features
- Summary statistics

🔍 Insights
- Gender: Females had a significantly higher survival rate.
- Class: First-class passengers were more likely to survive.
- Fare: Higher fare correlated positively with survival.
- Age: Most passengers were between 20–40 years old.
- Missing Data: Age and deck columns had substantial missing values.

🛠 Tools & Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
📎 Dataset Source
Titanic: Machine Learning from Disaster


Code:

#step 1 import required Libaries
%pip install seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step 2 Load dataset
df = sns.load_dataset('titanic')

# Step 3: Show head of your data
display(df.head())

# Step 4: Show info of your data
df.info()

# Step 5: Missing values
missing_values = df.isnull().sum()
print(missing_values)

# Step 6: Visualize age distribution
plt.figure(figsize=(5, 5))
sns.histplot(df['age'].dropna(), kde=True, bins=30, color='green')
plt.title("Age Distribution of Titanic dataset")
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Step 7: Survivor count by gender
plt.figure(figsize=(5, 5))
sns.countplot(x='sex', hue='survived', data=df)
plt.title("Survivor count by gender")
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Step 8: Survivor count by class
plt.figure(figsize=(5, 5))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival count by passenger class")
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Step 9: Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 10: Summary
print("Summary of Titanic dataset")
print(df.describe())

OUTPUT

Note: you may need to restart the kernel to use updated packages.Requirement already satisfied: seaborn in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (0.13.2)
Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from seaborn) (3.6.2)
Requirement already satisfied: pandas>=1.2 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from seaborn) (1.5.1)
Requirement already satisfied: numpy!=1.24.0,>=1.20 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from seaborn) (1.23.5)
Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.4)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.38.0)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.0.6)
Requirement already satisfied: pillow>=6.2.0 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (9.3.0)
Requirement already satisfied: pyparsing>=2.2.1 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.0.9)
Requirement already satisfied: cycler>=0.10 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.11.0)
Requirement already satisfied: packaging>=20.0 in c:\users\chara\appdata\roaming\python\python310\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (25.0)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from pandas>=1.2->seaborn) (2022.6)
Requirement already satisfied: six>=1.5 in c:\users\chara\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)


[notice] A new release of pip available: 22.3.1 -> 25.2
[notice] To update, run: python.exe -m pip install --upgrade pip

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     891 non-null    int64   
 1   pclass       891 non-null    int64   
 2   sex          891 non-null    object  
 3   age          714 non-null    float64 
 4   sibsp        891 non-null    int64   
 5   parch        891 non-null    int64   
 6   fare         891 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        891 non-null    category
 9   who          891 non-null    object  
 10  adult_male   891 non-null    bool    
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object  
 13  alive        891 non-null    object  
 14  alone        891 non-null    bool    
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64

<Figure size 500x500 with 1 Axes><img width="454" height="470" alt="image" src="https://github.com/user-attachments/assets/4b4a063e-efa7-4186-84bb-bf9c31dbf494" />
 
<Figure size 500x500 with 1 Axes><img width="463" height="470" alt="image" src="https://github.com/user-attachments/assets/9020a8bf-cc07-4686-a9a5-2d2ef685d625" />

<Figure size 500x500 with 1 Axes><img width="463" height="470" alt="image" src="https://github.com/user-attachments/assets/c74d1488-b28f-43bf-86ce-c1f81b55f4bf" />
 
<Figure size 800x600 with 2 Axes><img width="637" height="528" alt="image" src="https://github.com/user-attachments/assets/9ab1d489-4251-4a33-8efc-84e92e526a56" />

 
 
 
 Summary of Titanic dataset
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200


