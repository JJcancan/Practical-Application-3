
# Practical Application III

The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

## Business Understanding
Business Goal: Develop a classification model to accurately predict if a client will subscribe to a term deposit from the bank given metrics gathered.

## Data Understanding 
There are 16 features and one target feature labeled "y" with a total 45211 number of rows. There are seven numeric, six categorical, and three boolean features with the target variable being a boolean "yes or no" value.
No column headers were changed. Ydataprofiler was used to create a generalized analysis of the features. 

Figure below shows the features with missing entries.

![Missing Bar Graph](https://github.com/user-attachments/assets/aa331aa6-48d7-46f9-8b65-3cc3bf8d83b6)

Education, poutcome, and contact features presented multiple rows of missing values. The figure below shows the correlation matrix of the all the data (features and target variable).

![Correlation Matrix (Understanding Dataset)](https://github.com/user-attachments/assets/bcff6e46-fc76-4cc4-8b0c-2c02081f3bbc)

Education and jobs have a strong correlation with each other so I considered dropping Education feature due to it's amount of missing values. "poutcome" and "contact" were considered to be dropped as well due to a majority of its rows being missing.
"previous" column highly skewed scaler preprocessing may help. After removing certain features, I removed rows that contained missing values. After cleaning the dataset, 13 features were left with with a total of 44923 rows.

## Engineering Features
A column transformer was made using a one-hot encoder for the categorical features and a standard scaler for numeric values. A label encoder was used for the target variable to convert the classes "yes" and "no" to boolean 1 and 0.
The features were expanded to 39 columns, this may slow down model runtimes so feature importance analysis may be needed. An 80/20 train test split was then performed to gather test and training datasets.

## Baseline Modelling
A sklearn dummy classifier was used to calculate baseline performance for the data. Below are the results

Baseline Classifier Accuracy: 0.7924318308291597

Classification Report:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Yes</th>
      <td>0.881003</td>
      <td>0.884664</td>
      <td>0.882830</td>
      <td>7942.000000</td>
    </tr>
    <tr>
      <th>No</th>
      <td>0.093069</td>
      <td>0.090125</td>
      <td>0.091573</td>
      <td>1043.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.792432</td>
      <td>0.792432</td>
      <td>0.792432</td>
      <td>0.792432</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.487036</td>
      <td>0.487394</td>
      <td>0.487201</td>
      <td>8985.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.789538</td>
      <td>0.792432</td>
      <td>0.790979</td>
      <td>8985.000000</td>
    </tr>
  </tbody>
</table>
</div>

Precision is fairly high in the baseline but for the business goal I want to focus on improving accuracy.

## Modelling
### Multiple Baseline Model Comparison
Instead of assessing the logistic regression model first I combined all of the model assessments in this section. I constructed a call to each model (logistic regression, KNN, Decision Tree, and SVC) with default parameters
to gather a first pass assessment.
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Training ACC</th>
      <th>Test ACC</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.892593</td>
      <td>0.895715</td>
      <td>0.264296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>1.000000</td>
      <td>0.868336</td>
      <td>0.937949</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.920947</td>
      <td>0.891931</td>
      <td>0.012965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM</td>
      <td>0.910930</td>
      <td>0.900612</td>
      <td>52.291974</td>
    </tr>
  </tbody>
</table>
</div>

Decision tree model had the lowest testing accuracy with SVC having the highest but the longest run time. KNN and logistic regression performed fairly similar in testing accuracy but with different training accuracy and runtime performances.

### Improving models

I wanted to do another passthrough analysis of all the models in comparison using GridSearchCV with five kfolds to fine tune the hyperparameters with a focus on accuracy. I ran the models using all features first, which I found computational extensive with hyperparameter tuning mostly for the SVC model (which is as expected).
I decided to omit SVC from the first run as the runtime became too long.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Training ACC</th>
      <th>Test ACC</th>
      <th>best_score</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>0.892621</td>
      <td>0.895826</td>
      <td>0.892036</td>
      <td>5.938163</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>0.912850</td>
      <td>0.899833</td>
      <td>0.895765</td>
      <td>92.167382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>0.913267</td>
      <td>0.894602</td>
      <td>0.891953</td>
      <td>49.021958</td>
    </tr>
  </tbody>
</table>
</div>

To include SVC, feature importance was done using a gradient booster classifier. Focusing on the more influential features can lower runtime and focus our model on variables that describe the target variable heavily.

![feature_importance](https://github.com/user-attachments/assets/b7173797-5386-474a-8b6f-9e62366e35b5)

The top 10 features frome the feature importance analysis was then used to run all models.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Best Parameters</th>
      <th>Training ACC</th>
      <th>Test ACC</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>{'C': 1}</td>
      <td>0.892092</td>
      <td>0.893600</td>
      <td>2.185155</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree</td>
      <td>{'criterion': 'gini', 'max_depth': 8, 'min_sam...</td>
      <td>0.910290</td>
      <td>0.897718</td>
      <td>38.477083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNN</td>
      <td>{'n_neighbors': 9, 'weights': 'uniform'}</td>
      <td>0.912766</td>
      <td>0.891597</td>
      <td>38.295566</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM</td>
      <td>{'C': 10, 'kernel': 'rbf'}</td>
      <td>0.911347</td>
      <td>0.901948</td>
      <td>966.425673</td>
    </tr>
  </tbody>
</table>
</div>

I was able to run SVC with the reduced dataset. Out of all models with hyperparameter tuning, Decision Tree model was the best performing with a reasonable run-time. 

### Model Assessment

The ROC curve for the decision tree model with the best parameters had a large are but I wonder what I could have done to increase this area.

![ROC_curve_10features](https://github.com/user-attachments/assets/a155713e-4558-43b4-b0bb-5d6555c28061)

The confusion matrix results shows how the model was focused on accuracy. The True Positive prediction is crucial to directing marketing campaigns as the client would look for targeting customers who are more likely to say yes to a term deposit subscription.

![Confusion_Matrix](https://github.com/user-attachments/assets/366bcc8a-7afe-4f48-ab86-a67a5fc35e8e)

### Conclusion

The best model was found using a Decision Tree model with the best parameters found ({'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 5}) using five kfolds Gridsearchcv hyperparameter tuning with a focus on accuracy.
The model was 10% more accurate than the dummy baseline model that was used as a first assessment. There could be better ways/models out there that could have better results but with the models given, this is the best method.
The model was focused on accuracy so that the client/bank can predict how marketing campaigns will do with true positives or customers that say yes to the subscription.





