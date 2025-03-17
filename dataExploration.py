import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up visualization style
plt.style.use('ggplot')
sns.set(style="white")

# Load the dataset
file_path = "TelecomCustomerChurn.csv" 
df = pd.read_csv(file_path)

# Print column names to see the exact case
print("Column names in the dataset:")
print(df.columns.tolist())

# Preprocess data
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Convert yes/no to 1/0 for Churn
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 1. Make a graph of the distribution of Churn in our input dataset
plt.figure(figsize=(10, 5))
ax = sns.countplot(x=df['Churn'])
plt.title("Distribution of Churn in the Dataset")
plt.xlabel("Churn (No=0, Yes=1)")
plt.ylabel("Count")

# Add count labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig("churnDistribution.jpg", dpi=300)
plt.close()

# 2. Convert categorical columns for correlation analysis
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = df.copy()

for col in categorical_cols:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# 3. Create correlation heatmap
corr_matrix = df_encoded.corr()
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.jpg', dpi=300)
plt.close()

# 4. Print top correlations with Churn
churn_corr = corr_matrix['Churn'].sort_values(ascending=False)
print("Top Feature Correlations with Churn:")
print(churn_corr)

# 5. Analyze contract type and tenure relationship with churn
plt.figure(figsize=(12, 6))
contract_churn = pd.crosstab(df['Contract'], df['Churn'])
contract_churn.plot(kind='bar', stacked=True)
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('contract_churn.jpg', dpi=300)
plt.close()

# 6. Analyze internet service and tech support relationship with churn
tech_internet_churn = pd.crosstab([df['InternetService'], df['TechSupport']], df['Churn'])
fig, ax = plt.subplots(figsize=(12, 8))
tech_internet_churn.plot(kind='bar', stacked=True, ax=ax)
plt.title('Churn by Internet Service and Tech Support')
plt.xlabel('Internet Service - Tech Support')
plt.ylabel('Count')
plt.legend(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('internet_tech_churn.jpg', dpi=300)
plt.close()

# 7. Tenure distribution by churn status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='Tenure', data=df)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Churn')
plt.ylabel('Tenure (months)')
plt.tight_layout()
plt.savefig('tenure_churn.jpg', dpi=300)
plt.close()

# 8. Monthly charges distribution by churn status
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')
plt.tight_layout()
plt.savefig('monthly_charges_churn.jpg', dpi=300)
plt.close()

# 9. Payment Method and Paperless Billing relationship with churn
payment_paperless_churn = pd.crosstab([df['PaymentMethod'], df['PaperlessBilling']], df['Churn'])
fig, ax = plt.subplots(figsize=(14, 8))
payment_paperless_churn.plot(kind='bar', stacked=True, ax=ax)
plt.title('Churn by Payment Method and Paperless Billing')
plt.xlabel('Payment Method - Paperless Billing')
plt.ylabel('Count')
plt.legend(['No Churn', 'Churn'])
plt.tight_layout()
plt.savefig('payment_paperless_churn.jpg', dpi=300)
plt.close()

print("Data exploration complete. All visualizations saved to files.")
