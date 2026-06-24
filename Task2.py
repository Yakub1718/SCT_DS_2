import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------------------------------
# Data Cleaning
# -------------------------------------------------------

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

print("\nAfter Cleaning:")
print(df.isnull().sum())

# -------------------------------------------------------
# CHART 1: Survival Count
# -------------------------------------------------------
plt.figure(figsize=(6,5))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count', fontsize=14, fontweight='bold')
plt.xticks([0,1], ['Not Survived', 'Survived'])
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart1_survival_count.png')
plt.show()

# -------------------------------------------------------
# CHART 2: Survival by Gender
# -------------------------------------------------------
plt.figure(figsize=(6,5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Gender', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart2_survival_gender.png')
plt.show()

# -------------------------------------------------------
# CHART 3: Survival by Passenger Class
# -------------------------------------------------------
plt.figure(figsize=(6,5))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Passenger Class', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart3_survival_class.png')
plt.show()

# -------------------------------------------------------
# CHART 4: Age Distribution
# -------------------------------------------------------
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart4_age_distribution.png')
plt.show()

# -------------------------------------------------------
# CHART 5: Correlation Heatmap
# -------------------------------------------------------
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart5_heatmap.png')
plt.show()

print("\n All charts saved successfully!")