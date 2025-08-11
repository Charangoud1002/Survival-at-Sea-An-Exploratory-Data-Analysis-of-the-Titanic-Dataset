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

