import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
Y = bank_marketing.data.targets 
  
# fetch dataset
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
Y = bank_marketing.data.targets 
V = bank_marketing.variables
  

#check each feature for nulls
null_X = pd.DataFrame({'Number of Nulls': X.isnull().sum(),
                       'Percentage': (X.isnull().sum() / len(X)) * 100})

##Evaluate features that have many nulls
#nulls are in categorical features which is fine - not all customers had education data, 'poutcome' which is a marketing campaign, or 'contact' a communication type filled out
#for one hot encoding these categorical features, NaNs won't matter anyways
#these columns may require different weighting in models once I transform the X dataset - unlikely because categorical
null_columns = ['education', 'contact', 'poutcome']
for column in null_columns:
    counts = X[column].value_counts(dropna=False)
    percentages = (X[column].value_counts(dropna=False, normalize=True)*100)
    df = pd.DataFrame({'Counts': counts, 'Percentage': percentages})
    print(f"\nColumn: {column}\n")
    print(df)


###transforming the dataset
#convert categorical features into multiple binary columns - one-hot encoding
categorical_columns = V[V['type'] == 'Categorical']['name'].tolist()
Transformed_Features = pd.get_dummies(X[categorical_columns])
Transformed_Features = Transformed_Features.astype(int)
#convert binary columns from True/False to 1 and 0
binary_columns = V[V['type']=='Binary']['name'].tolist()
mapping_dict = {'yes': 1, 'no': 0}
for col in binary_columns:
    if col in X.columns:
        X.loc[:, col] = X[col].replace(mapping_dict)
#Though not relevant for a decision tree, i'm not sure which model to use yet so first check the distribution of values for continous features, if guassian, i'd conduct a standard min max noramlization, otherwise take the log of each number in order to normalize in case I use logit regression
import matplotlib.pyplot as plt

# Select integer type columns
df_int = X.select_dtypes(include=['int', 'int64'])

# plot histograms
df_int.hist(bins=50, figsize=(20,15))
plt.tight_layout()  # optional, to avoid overlapping of sub-plots
plt.show()

#because I'm not familiar with this data set, I'll likely end up using a decision tree based model as opposed to finding the most appropriate form of normalization for each feature

#date variables for opening bank accounts likely are contextual to the macro-economic environment or greater industry trends or business decisions
#not going to one-hot encode, I could take sin(2* np.pi), I could run a logistic regression between the months and the target datatable to see if there's a relationship, but for this exercise I'm taking date out because intuitively this isn't a feature I want in a bank marketing model
final_df.drop(['month','day_of_week'],axis=1, inplace=True)

##merge all tables into Transformed_Features table
# Drop categorical columns from X they're already in Transformed_Features
X = X.drop(columns=categorical_columns)
final_df = pd.merge(X, Transformed_Features, left_index=True, right_index=True)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# split the dataframe into features and target
features = final_df
target = Y['y']


# split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.30, random_state=42)

n_estimators_options = [100, 200, 500]
max_features_options = ['sqrt', 'log2']
max_depth_options = [4, 6, 8]
criterion_options = ['gini', 'entropy']

best_score = 0
best_params = {}

for n_estimators in n_estimators_options:
    for max_features in max_features_options:
        for max_depth in max_depth_options:
            for criterion in criterion_options:
                rfc = RandomForestClassifier(n_estimators=n_estimators,
                                             max_features=max_features,
                                             max_depth=max_depth,
                                             criterion=criterion,
                                             random_state=42)
                rfc.fit(X_train, Y_train)
                                # Use 5-fold CV, as an example. Adjust the number according to your preferences
                scores = cross_val_score(rfc, features, target, cv=5)

                # score is the average accuracy across the folds
                score = scores.mean()
                if score > best_score:
                    best_score = score
                    best_params = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'criterion': criterion}
                    
importances = rfc.feature_importances_
# Convert the importances into a DataFrame
importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values('importance', ascending=False)

print(importances_df)
print('Area Under the Curve:', best_score)
print('Best parameters:', best_params)






