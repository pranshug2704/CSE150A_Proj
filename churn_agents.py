import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ChurnExplanationAgent:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = {}  # Decision tree will be stored here
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        df = self.load_data(self.data_path)
        df = self.preprocess_data(df)

        # Separate features and target variable
        X = df.drop(['customerID', 'Churn'], axis=1)
        y = df['Churn']

        # Balance dataset (oversampling)
        X, y = self.oversample(X, y)

        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(X, y, test_size=0.2)

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def preprocess_data(self, df):
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()

        # Convert categorical variables to numerical values
        for col in df.select_dtypes(include=['object']).columns:
            df.loc[:, col] = df[col].astype('category').cat.codes

        return df

    def oversample(self, X, y):
        """ Custom oversampling function to balance dataset. """
        unique, counts = np.unique(y, return_counts=True)
        max_count = max(counts)

        X_balanced = []
        y_balanced = []

        for label in unique:
            indices = np.where(y == label)[0]
            oversampled_indices = np.random.choice(indices, max_count, replace=True)
            X_balanced.append(X.iloc[oversampled_indices])
            y_balanced.append(y.iloc[oversampled_indices])

        X_balanced = pd.concat(X_balanced, axis=0).reset_index(drop=True)
        y_balanced = np.concatenate(y_balanced)

        return X_balanced, y_balanced

    def train_test_split(self, X, y, test_size=0.2):
        """ Custom function to split data into training and testing sets. """
        split_index = int(len(X) * (1 - test_size))
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    def train_model(self):
        """ Trains a simple Decision Tree model using conditional probability tables (CPTs). """
        features = self.X_train.columns
        self.model = {feature: self.compute_cpt(self.X_train, feature, self.y_train) for feature in features}

    def compute_cpt(self, X, parent, y):
        """ Compute Conditional Probability Tables (CPTs) for a feature and target variable. """
        df = X.copy()
        df['Churn'] = y
        cpt = df.groupby([parent, 'Churn']).size().unstack().fillna(0)
        cpt = cpt.div(cpt.sum(axis=1), axis=0)  # Normalize probabilities
        return cpt

    def explain_churn(self):
        """ Print simplified decision rules based on computed CPTs. """
        print("Decision Rules Explaining Churn:")
        for feature, cpt in self.model.items():
            print(f"\nFeature: {feature}")
            print(cpt)

    def predict_churn(self, customer):
        """ Compute churn probability for a given customer profile using CPTs and Bayes' rule. """
        # Prior probabilities (can be computed from training data)
        p_churn_prior = 0.27  # Approximate churn rate in the dataset
        p_not_churn_prior = 0.73

        # Likelihood computation: P(Features|Churn)
        p_features_given_churn = 1.0 
        p_features_given_not_churn = 1.0 

        for feature, val in customer.items():
            if feature in self.model:
                cpt = self.model[feature]
                if val in cpt.index:
                    # Multiply by likelihood of each feature given churn status
                    p_features_given_churn *= cpt.loc[val, 1]
                    p_features_given_not_churn *= cpt.loc[val, 0]

        # Apply Bayes' rule: P(Churn|Features) = P(Features|Churn) * P(Churn) / P(Features)
        # We don't need to compute P(Features) explicitly since we'll normalize
        p_churn_unnormalized = p_features_given_churn * p_churn_prior
        p_not_churn_unnormalized = p_features_given_not_churn * p_not_churn_prior

        # Normalize to get probabilities
        total = p_churn_unnormalized + p_not_churn_unnormalized
        p_churn_final = p_churn_unnormalized / total
        p_not_churn_final = p_not_churn_unnormalized / total

        print("\nExample Customer: ")
        print(f"P(Churn) = {p_churn_final:.4f}")
        print(f"P(No Churn) = {p_not_churn_final:.4f}")
        
        # Return prediction
        return 1 if p_churn_final > p_not_churn_final else 0

    def evaluate_model(self):
        """Evaluate the model on the test set and report metrics."""
        # Make predictions on test set
        y_pred = []
        for _, row in self.X_test.iterrows():
            customer = row.to_dict()
            pred = self.predict_churn(customer)
            y_pred.append(pred)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='binary')
        recall = recall_score(self.y_test, y_pred, average='binary')
        f1 = f1_score(self.y_test, y_pred, average='binary')
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1

if __name__ == "__main__":
    agent = ChurnExplanationAgent('TelecomCustomerChurn.csv')
    agent.train_model()
    
    # Print example CPTs to understand the model
    print("Example Conditional Probability Tables:")
    if "Contract" in agent.model:
        print("\nCPT for Contract:")
        print(agent.model["Contract"])
    if "tenure" in agent.model:
        print("\nCPT for tenure:")
        print(agent.model["tenure"])
    
    # Evaluate model performance
    agent.evaluate_model()
    
    # Example customer profile
    customer_example = {
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 1, 
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "Contract": 0,  # Month-to-month contract
        "PaperlessBilling": 1,
        "PaymentMethod": 1,
        "MonthlyCharges": 2,
        "TotalCharges": 1
    }

    print("\nPrediction for example customer:")
    agent.predict_churn(customer_example)
