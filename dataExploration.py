import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "TelecomCustomerChurn.csv" 
df = pd.read_csv(file_path)

# Make a graph of the distribution of Churn in our input dataset
plt.figure(figsize=(10, 5))
sns.countplot(x=df['Churn'])
plt.title("Distribution of Churn in the Dataset")
plt.xlabel("Churn (No=0, Yes=1)")
plt.ylabel("Count")
plt.show()
