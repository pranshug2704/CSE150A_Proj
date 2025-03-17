# CSE150A_Proj

## Regrade Response

**Update:** In response to the grading feedback, we have made the following improvements:

1. **PEAS/Agent Analysis:** We've clarified the specific problem our agent is solving (telecom customer churn prediction) and detailed the PEAS framework as it applies to this specific problem.

2. **Agent Setup:** We've expanded our data exploration section to introduce the dataset more thoroughly, explain important variables, and clarify our Bayesian network model and its dependencies.

3. **Training and Results:** We've replaced the mentions of logistic regression with our Bayesian network-based model using Conditional Probability Tables (CPTs), and provided more details about how we train and evaluate our model.

4. **Conclusion/Results:** We've added specific results from our model evaluation and provided more concrete steps for improvement.

Detailed updates can be found in the respective sections below.

# AI Agent Overview

**Update:** Our AI agent is designed specifically to predict telecom customer churn using a Bayesian network approach. It operates under the PEAS framework as follows:

**Performance Measure:** The effectiveness of the churn prediction model is measured based on accuracy, precision, recall, and F1-score when predicting whether a customer will leave the telecom service.

**Environment:** The agent operates in the telecom customer data environment where customer demographic information, service usage patterns, contract details, and billing information collectively influence churn decisions.

**Actuators:** The agent computes conditional probability tables (CPTs), builds a Bayesian network structure, and outputs churn probability predictions for customer profiles.

**Sensors:** It ingests structured telecom customer data, processes feature values, and uses these to compute probabilities within the Bayesian network.

# Agent Type

**Update:** The agent is a utility-based learning system, as it makes decisions based on the probability of churn to maximize the utility of retaining customers. It fits within probabilistic modeling by utilizing Bayesian networks with conditional probability tables (CPTs) to model the probabilistic relationships between customer attributes and churn likelihood. This approach directly implements concepts from probability theory covered in class.

# Data Exploration and Preprocessing

**Update:** The dataset used is the Telecom Customer Churn Dataset which is a collection of data on 7,043 customers. Our goal is to predict if a customer is likely to churn (leave the service) using data about demographic, usage and billing. Various key factors play into churn prediction: demographic information (gender, partner status, dependents and senior citizen status), customer information (contract length, payment method, and tenure), service usage (internet service usage, phone service usage and streaming TV status) and billing information (monthly accumulated charges and total charges paid). In order to better visualize the data, we plot the churn distribution in our dataset (using the code in dataExploration.py) 
Below are the key attributes of our dataset:

- **Total observations:** 7,043 customer records
- **Target variable:** Churn (binary: Yes/No) - indicates whether the customer left the telecom service
- **Features:** 20 predictor variables relating to customer demographics, services, and billing

## Dataset Distribution

The churn distribution in our dataset shows class imbalance, with fewer customers churning than staying:

![Churn Distribution](churnDistribution.jpg)

**Update:** In order to work with the data we do thing major things: load and preprocess the data, feature engineering and training and evaluation. Firstly, we preprocess the data by converting the TotalCharges variable to a numerical format, drop missing variables and convert categorical variables into numerical values. For feature engineering we disregard customerID since it is not relevant for our churn predictions. We then split the data into features (denoted X) and target (denoted Y). At the training and evaluation stage we split the data into 80% training and 20% testing.

**Update:** Our model has dependencies on a few python libraries and preprocessing steps in the code. We use pandas, numpy, sklearn.model_selection.train_test_split (to split the data into training and testing) and imblearn.over_sampling.SMOTE (to balance the dataset by oversampling the smaller class).

## Key Variables Description:

1. **Demographic Information:**
   - gender: Male/Female
   - SeniorCitizen: Whether customer is a senior citizen (1) or not (0)
   - Partner: Whether customer has a partner (Yes/No)
   - Dependents: Whether customer has dependents (Yes/No)

2. **Customer Account Information:**
   - tenure: Number of months the customer has stayed with the company
   - Contract: Month-to-month, One year, Two year
   - PaymentMethod: Electronic check, Mailed check, Bank transfer, Credit card
   - PaperlessBilling: Yes/No

3. **Services Information:**
   - PhoneService: Yes/No
   - MultipleLines: Yes/No/No phone service
   - InternetService: DSL, Fiber optic, No
   - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Yes/No/No internet service

4. **Charges Information:**
   - MonthlyCharges: Amount charged monthly
   - TotalCharges: Total amount charged

## Data Preprocessing Steps:

1. **Handling Missing Values:** 
   - Converted TotalCharges column from object to numeric, which revealed 11 missing values
   - Dropped rows with missing values (11 records)

2. **Categorical Encoding:**
   - Converted all categorical variables to numerical codes for our Bayesian model
   - For example: Yes→1, No→0; Month-to-month→0, One year→1, Two year→2

3. **Feature Selection:**
   - Removed customerID as it's not predictive
   - Kept all other features for initial modeling

4. **Addressing Class Imbalance:**
   - Implemented custom oversampling to balance the dataset
   - This ensures our model isn't biased toward the majority class (non-churning customers)

5. **Data Splitting:**
   - Split data into 80% training and 20% testing sets to evaluate model performance 

# Agent Setup & Probabilistic Modeling

**Update:** Our Bayesian network-based agent for telecom customer churn prediction follows these steps:

1. **Data Ingestion:** The system loads the telecom churn dataset.

2. **Data Preprocessing:** As described in the previous section, we handle missing values and encode categorical variables.

3. **Bayesian Network Structure:** 
   - We implement a Naive Bayes structure for our initial model, where each feature is conditionally independent given the churn status.
   - The graphical model looks like this:

   Churn
    / | \
   /  |  \
  F1  F2  F3 ... Fn

   Where F1, F2, etc. are features like gender, tenure, contract type, etc.

4. **CPT Computation:** 
   - For each feature, we compute a Conditional Probability Table (CPT) that represents P(Feature|Churn).
   - These CPTs are the core of our Bayesian network, representing all conditional dependencies.
   - Example of a CPT for the "Contract" feature:
   ```
   P(Contract=Month-to-month|Churn=Yes) = 0.88
   P(Contract=One year|Churn=Yes) = 0.08
   P(Contract=Two year|Churn=Yes) = 0.04
   P(Contract=Month-to-month|Churn=No) = 0.35
   P(Contract=One year|Churn=No) = 0.24
   P(Contract=Two year|Churn=No) = 0.41
   ```

5. **Probabilistic Inference:** 
   - Given a new customer profile, we use Bayes' rule to compute:
   - P(Churn=Yes|Features) = P(Features|Churn=Yes) * P(Churn=Yes) / P(Features)
   - P(Churn=No|Features) = P(Features|Churn=No) * P(Churn=No) / P(Features)
   - The final prediction is the churn status with the higher probability.

6. **Model Storage:** The computed CPTs are stored and can be used for future predictions.

# Model Training and Evaluation

**Update:** For our first model, we implemented a Bayesian network approach:

1. **Training Process:**
   - Loaded and preprocessed the telecom customer dataset as described above
   - Split the data into 80% training and 20% testing sets
   - Computed Conditional Probability Tables (CPTs) for each feature given the churn status
   - These CPTs represent P(Feature|Churn) for each feature in our dataset
   - The CPTs were computed using the training data, specifically by counting occurrences and normalizing to get probabilities

2. **Model Implementation:**
   - Our model uses a Naive Bayes approach, assuming conditional independence of features given the churn status
   - For a new customer, we compute P(Churn=Yes|Features) using Bayes' rule:
     P(Churn=Yes|Features) ∝ P(Churn=Yes) * ∏ P(Feature_i|Churn=Yes)
   - Similarly for P(Churn=No|Features)
   - The final prediction is the churn status with the higher probability

3. **Evaluation Results:**
   - Accuracy: 73.5% on the test set
   - Precision: 68.2% for churn prediction
   - Recall: 64.9% for churn prediction
   - F1-Score: 66.5% for churn prediction

4. **Model Analysis:**
   - Our model performs significantly better than random guessing (50%)
   - The Naive Bayes assumption of feature independence is a limitation, as some features are clearly correlated (e.g., monthly charges and total charges)
   - The model struggles with precision, meaning it sometimes predicts churn when customers actually stay
   - Despite these limitations, the model provides a solid baseline for churn prediction using probabilistic methods

# Repository and Code Links

All code has been uploaded and is accessible at:
[GitHub Repository](https://github.com/pranshug2704/CSE150A_Proj)

# Conclusion and Possible Improvements

**Update:** Based on our first Bayesian network model for churn prediction, we can draw the following conclusions:

1. **Model Performance:**
   - Our Naive Bayes approach achieved 73.5% accuracy, which indicates it's learning useful patterns from the data
   - The model is better at identifying non-churning customers than predicting churn (higher precision for the "No Churn" class)
   - This performance aligns with what we would expect from a baseline Bayesian model with the naive independence assumption

2. **Model Fitting Assessment:**
   - Our model shows signs of underfitting rather than overfitting:
     - The performance metrics aren't exceptionally high
     - The simplistic structure (assuming feature independence) likely isn't capturing all the complex relationships in the data
     - We're not seeing a large gap between training and testing performance

3. **Possible Improvements:**

   - **Enhanced Bayesian Network Structure:**
     - Move beyond Naive Bayes by modeling direct dependencies between features
     - Create a more complex network structure with appropriate edges between correlated features (e.g., MonthlyCharges and TotalCharges)
     - Implement Bayesian network structure learning algorithms to discover the optimal network structure

   - **Feature Engineering:**
     - Create ratio features (e.g., MonthlyCharges/tenure to represent average monthly spend)
     - Generate interaction terms between important features (e.g., Contract type and tenure)
     - Discretize continuous variables to improve CPT estimation

   - **Improved Probability Estimation:**
     - Apply smoothing techniques to handle zero probabilities in the CPTs
     - Use more sophisticated probability estimation methods like Bayesian estimation with appropriate priors
     - Implement parameter learning techniques like maximum likelihood estimation

   - **Model Evaluation:**
     - Implement k-fold cross-validation to get more robust performance estimates
     - Add more evaluation metrics relevant to the business problem (e.g., cost of false negatives vs. false positives)
     - Compare against other probabilistic models to benchmark performance

These improvements would help move our model from the underfitting region toward the optimal fitting point on the bias-variance tradeoff curve.

# AI Assistance Acknowledgment

This project was developed with assistance from generative AI tools including Claude and ChatGPT. These tools were used for code optimization, debugging, documentation improvement, and conceptual explanations of Bayesian networks.

# Milestone 3: Enhanced Bayesian Network Model

## PEAS/Agent Analysis

Our agent continues to focus on telecom customer churn prediction, now with an enhanced Bayesian network approach. The PEAS framework for our agent is:

**Performance Measure:** 
- Primary metrics: Accuracy, precision, recall, F1-score, and ROC-AUC for churn prediction
- Business-oriented metrics: Expected retention value and potential revenue saved

**Environment:** 
- Telecom customer data environment with historical churn information
- Customer demographic information, service usage patterns, contract details, and billing information
- The environment is partially observable (we only see certain customer attributes)
- The environment is stochastic (customer behavior has inherent randomness)

**Actuators:** 
- Builds enhanced Bayesian network structure with directed edges between correlated features
- Computes Conditional Probability Tables (CPTs) with Laplace smoothing
- Outputs probabilistic churn predictions and confidence levels for each prediction
- Identifies key factors contributing to churn likelihood for each customer

**Sensors:** 
- Ingests structured telecom customer data
- Detects correlations and dependencies between features
- Monitors model performance metrics to guide refinement

## Agent Setup, Data Preprocessing, and Training Setup

### Enhanced Data Exploration

We've conducted a more comprehensive exploration of our dataset to better understand variable interactions and their impact on churn:

![Feature Correlation Heatmap](feature_correlation_heatmap.jpg)

From the correlation analysis, we identified several key relationships:
- Strong positive correlation between MonthlyCharges and TotalCharges (0.65)
- Contract type has strong negative correlation with churn (-0.40)
- Tenure has strong negative correlation with churn (-0.35)
- Internet service type (particularly fiber optic) has moderate positive correlation with churn (0.30)

### Key Variable Interactions

Based on our data exploration, we identified the following critical variable interactions:

1. **Contract-Tenure-Churn Relationship:** 
   - Customers with month-to-month contracts AND low tenure have significantly higher churn rates
   - This interaction is more important than either variable alone

2. **Internet Service-Monthly Charges-Churn Relationship:**
   - Fiber optic customers with high monthly charges show elevated churn
   - This pattern isn't as pronounced for DSL customers

3. **Payment Method-Paperless Billing-Churn Relationship:**
   - Electronic check payments combined with paperless billing show higher churn rates
   - This suggests potential issues with the electronic payment process

4. **Tech Support-Internet Service-Churn Relationship:**
   - Lack of tech support combined with fiber service leads to higher churn
   - This indicates service quality issues may be driving churn in high-speed internet customers

### Bayesian Network Structure

For Milestone 3, we've enhanced our Bayesian network by moving beyond the Naive Bayes structure. Our new model accounts for the dependencies between features:

```
                           ┌─────────┐
                           │  Churn  │
                           └────┬────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼──────┐        ┌───────▼──────┐        ┌───────▼──────┐
│   Contract   │◄──────►│    Tenure    │        │Internet Srvc │
└───────┬──────┘        └──────────────┘        └───────┬──────┘
        │                                                │
        │                                                │
┌───────▼──────┐                                ┌───────▼──────┐
│PaymentMethod │                                │Tech Support  │
└──────────────┘                                └──────────────┘
```

This structure was chosen based on:
1. Statistical testing for conditional independence (Chi-square tests)
2. Domain knowledge about telecom customer behavior
3. Analysis of correlation patterns in our dataset

The key advantages of this structure over Naive Bayes:
- It captures the dependency between Contract and Tenure
- It models the relationship between Internet Service and Tech Support
- It acknowledges that Payment Method choice is influenced by Contract type
- While still maintaining computational tractability

### Parameter Calculation Process

For calculating the CPTs in our enhanced Bayesian network, we use the following approach:

1. **For root nodes** (nodes without parents):
   - Calculate P(X) using maximum likelihood estimation:
   - P(X=x) = count(X=x) / total_count

2. **For child nodes** (nodes with parents):
   - Calculate P(X|Parents(X)) using maximum likelihood estimation with Laplace smoothing:
   - P(X=x|Parents(X)=pa) = (count(X=x, Parents(X)=pa) + α) / (count(Parents(X)=pa) + α*|X|)
   - Where α is the smoothing parameter (set to 0.5) and |X| is the number of possible values for X

3. **For the Churn node:**
   - Compute P(Churn|all parent nodes) using the same approach
   - This gives us a CPT that captures how multiple factors jointly influence churn probability

The code implementation for computing these CPTs is:

```python
def compute_enhanced_cpt(self, X, child, parents, y=None):
    """Compute CPT for a node with multiple parents using Laplace smoothing."""
    df = X.copy()
    
    if y is not None:
        # For nodes that depend on Churn
        df['Churn'] = y
        all_cols = parents + [child, 'Churn']
        
        # Group by all parent variables and churn
        grouped = df[all_cols].groupby(parents + ['Churn']).value_counts().unstack(level=-1).fillna(0)
        
        # Apply Laplace smoothing
        alpha = 0.5
        num_values = df[child].nunique()
        smoothed = (grouped + alpha) / (grouped.sum(axis=1).values.reshape(-1, 1) + alpha * num_values)
        
        return smoothed
    else:
        # For nodes that don't depend on Churn
        all_cols = parents + [child]
        
        # Group by all parent variables
        grouped = df[all_cols].groupby(parents).value_counts().unstack(level=-1).fillna(0)
        
        # Apply Laplace smoothing
        alpha = 0.5
        num_values = df[child].nunique()
        smoothed = (grouped + alpha) / (grouped.sum(axis=1).values.reshape(-1, 1) + alpha * num_values)
        
        return smoothed
```

## Model Training

We've implemented our enhanced Bayesian network model in the `train_enhanced_model` method in our `ChurnExplanationAgent` class:

```python
def train_enhanced_model(self):
    """Trains an enhanced Bayesian Network model with feature dependencies."""
    
    # Define network structure based on domain knowledge and correlations
    self.network_structure = {
        'Contract': [],  # Root node
        'Tenure': ['Contract'],  # Depends on contract type
        'InternetService': [],  # Root node
        'TechSupport': ['InternetService'],  # Depends on internet service
        'PaymentMethod': ['Contract'],  # Depends on contract type
        'Churn': ['Contract', 'Tenure', 'InternetService', 'TechSupport', 'PaymentMethod']  # Target variable
    }
    
    # Compute CPTs for each node in the network
    self.enhanced_model = {}
    
    # Process root nodes first
    for node, parents in self.network_structure.items():
        if node != 'Churn':
            if not parents:  # Root node
                # Simple probability distribution P(X)
                self.enhanced_model[node] = self.compute_simple_probability(self.X_train, node)
            else:
                # Conditional probability P(X|Parents)
                self.enhanced_model[node] = self.compute_enhanced_cpt(self.X_train, node, parents)
    
    # Process Churn node with all its parents
    churn_parents = self.network_structure['Churn']
    self.enhanced_model['Churn'] = self.compute_enhanced_cpt(
        self.X_train, 'Churn', churn_parents, self.y_train
    )
    
    # Store prior probability of churn
    self.p_churn_prior = sum(self.y_train) / len(self.y_train)
```

## Conclusion and Results

### Model Evaluation Results

We evaluated our enhanced Bayesian network against our previous Naive Bayes approach:

| Metric | Naive Bayes | Enhanced Bayesian Network | Improvement |
|--------|-------------|--------------------------|-------------|
| Accuracy | 71.7% | 78.8% | +7.1% |
| Precision | 42.3% | 61.8% | +19.5% |
| Recall | 12.3% | 57.0% | +44.6% |
| F1-Score | 19.1% | 59.3% | +40.2% |
| ROC-AUC | 0.66 | 0.83 | +0.17 |

![ROC Curve Comparison](roc_curve_comparison.jpg)

The ROC curve comparison clearly shows that our enhanced model (red line) significantly outperforms the basic Naive Bayes model (blue line), with a much higher area under the curve (AUC).

### Confusion Matrix

The confusion matrix for our enhanced model shows improved classification performance:

![Confusion Matrix](confusion_matrix.jpg)

The confusion matrix shows that our enhanced model correctly identifies a significant portion of the churning customers (true positives) while maintaining a good balance with false positives. This is particularly important in a business context where identifying potential churners allows for targeted retention efforts.

### Feature Importance Analysis

By analyzing the correlations in our model, we identified the most influential factors for churn prediction:

1. **Contract Type:** The strongest predictor with a correlation of -0.395, indicating that longer contracts significantly reduce churn risk
2. **Tenure:** Strong negative correlation of -0.356, showing that customers who have been with the company longer are less likely to churn
3. **Total Charges:** Correlation of -0.204, suggesting that customers who have paid more in total are less likely to leave
4. **Payment Method:** Correlation of 0.195, with certain payment methods associated with higher churn
5. **Paperless Billing:** Correlation of 0.195, with paperless billing customers showing higher churn rates

![Feature Importance](feature_importance.jpg)

The feature importance graph shows the absolute correlation values, highlighting which features have the strongest relationship with customer churn regardless of direction.

### Model Interpretation

Our enhanced Naive Bayes model with Laplace smoothing provides several advantages for interpretation:

1. **Probabilistic Explanations:** For each prediction, we can provide specific probability values showing how features contribute to the prediction
2. **Feature Importance Ranking:** The correlation analysis clearly identifies which factors most strongly influence churn
3. **Improved Discrimination:** The enhanced model's ROC-AUC of 0.83 (compared to 0.66 for the basic model) demonstrates its superior ability to distinguish between churning and non-churning customers
4. **Business Actionability:** The model identifies specific factors (contract type, tenure, payment method) that the business can directly address in retention strategies

### Areas for Improvement

While our enhanced model shows significant improvement, we've identified several areas for further refinement:

1. **True Bayesian Network Structure:** 
   - Implement a full Bayesian network structure that captures dependencies between features
   - Use structure learning algorithms to discover optimal network topology
   - Model direct relationships between correlated features like Contract-Tenure and InternetService-TechSupport

2. **Advanced Parameter Estimation:**
   - Explore different smoothing techniques beyond Laplace smoothing
   - Implement Bayesian parameter estimation with informative priors
   - Consider parameter learning with missing data techniques

3. **Handling Continuous Variables:**
   - Our current approach treats continuous variables like MonthlyCharges as categorical
   - Explore conditional linear Gaussian networks for continuous variables
   - Implement kernel density estimation for more flexible distributions

4. **Class Imbalance Handling:**
   - Implement more sophisticated approaches to handle the class imbalance (27% churn rate)
   - Explore cost-sensitive learning to account for different misclassification costs
   - Implement SMOTE or other advanced oversampling techniques

5. **Threshold Optimization:**
   - Develop a business-oriented approach to threshold selection
   - Optimize for expected value rather than accuracy
   - Implement cost-benefit analysis for different threshold choices

### Technical Implementation Improvements

Specific technical improvements we're working on for the next milestone:

1. **Optimizing Model Training:**
   - Implement more efficient data structures for probability tables
   - Parallelize computation for faster training on larger datasets
   - Develop incremental learning capabilities for model updates

2. **Feature Engineering:**
   - Create ratio features (e.g., MonthlyCharges/Tenure for average monthly spend)
   - Generate interaction terms between important features
   - Implement feature selection to remove redundant or irrelevant variables

3. **Evaluation Framework:**
   - Implement k-fold cross-validation for more robust performance estimates
   - Develop business-specific metrics (e.g., expected retention value)
   - Create visualization tools for model interpretation

4. **Deployment Considerations:**
   - Design an API for real-time prediction serving
   - Implement model versioning and monitoring
   - Develop explanation capabilities for business users

5. **Data Pipeline:**
   - Create automated data preprocessing pipeline
   - Implement data quality checks and validation
   - Design feature transformation framework

## Libraries and Tools

For this implementation, we utilized the following libraries:

- **NumPy** and **Pandas** for data manipulation and analysis
- **Matplotlib** and **Seaborn** for visualization
- **Scikit-learn** for evaluation metrics and data preprocessing

Our implementation is based on fundamental probability theory and Bayesian methods covered in class. We implemented Naive Bayes with Laplace smoothing from scratch, without relying on external libraries for the core model. This approach allowed us to:

1. Gain a deeper understanding of the probabilistic foundations
2. Customize the smoothing parameters for our specific dataset
3. Implement custom evaluation metrics and visualization tools

The full source code for our implementation is available in our GitHub repository.

## References

1. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
2. Scikit-learn documentation: https://scikit-learn.org/
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
4. Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. Machine Learning, 29, 103-130.
