# ML-Threat-Detection
Prediction of malicious network connection events with Random Forest and Elastic-Net Regression in R and Python

### * This report and plots are based on models developed in R.
### ** Similar models have been implemented in Python to accommodate different business requirements.
### *** In Python, certain libraries or methods used require one-hot encoding of categorical features. This process increases the dimensionality of the data and can lead to increased model complexity and potential overfitting, which may degrade model performance. As a result, the Python models showed slightly lower performance on certain metrics compared to the R models. However, they still exhibited similar trends and patterns as observed in the R models.

## 1. INTRODUCTION
This project aims to develop an early warning system for a cyber-security department to detect potential network attacks. Leveraging machine learning models, the goal is to classify real-time network events as either malicious or non-malicious. The challenge lies in the rarity of confirmed malicious events, accounting for less than 1% of all logged network events, and the need to minimize both false positives and negatives due to their significant implications. Two machine learning algorithms, Random Forest and Elastic-Net Regression will be tested and evaluated to determine the most accurate incident detection model.

## 2. BUSINESS USE 
This project directly addresses a critical issue faced by Cyber Security departments: the detection of potential network attacks in real-time. By enhancing existing Security Information and Event Management (SIEM) system with machine learning models, we aim to provide immediate classification of network events as malicious or non-malicious. This real-time feedback can significantly reduce the response time to potential threats, thereby enhancing cybersecurity defences.
 
The models developed in this project have broad applications in various sectors. In finance, similar models can be used for fraud detection, identifying suspicious transactions in real-time to prevent financial losses. In healthcare, these models can be used to detect anomalies in patient data, potentially identifying health issues before they become critical. In the field of IoT, such models can help secure devices and networks by identifying unusual patterns in data traffic. In essence, any industry dealing with large volumes of data and requiring real-time anomaly detection can benefit from the application of these machine learning models.

## 3. THE DATA 
The dataset used for this project contains 502159 observations of network connection events. It was decided to drop the IPV6.Traffic feature for having too many invalid values, so the dataset was left with 13 variables: a “Class” variable that determines if the event was confirmed as malicious or non-malicious, 3 categorical and 9 continuous features, including Operating System(source), Ingress Router, Packet Size and more. 
 
To try different training settings for each model, an UNBALANCED training set with 19800 non-malicious and 200 malicious samples, and a BALANCED training set with 19800 non-malicious and malicious samples each(malicious samples were bootstrapped to reach the desired quantity) were defined. The test set is composed by the remaining 472036 observations not used on the training sets. It is important to note that on the test set, only 2741 or 0.038% of the samples correspond to malicious events, demonstrating a great imbalance in the data. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20the%20data.png)

## 4. DATA CLEANING & PREPARATION 
Some of the features contained categories with low representation in the data. To reduce bias on the classification, simplify the dataset improving its interpretability and improve computational efficiency, the feature “Connection State” had its “INVALID, RELATED, NEW” categories merged into “Others” and the feature “Operating System” had its “OTHER, IOS, LINUX” categories also merged into “Others”. The “IPV6 Traffic” feature was removed from the set for containing a large number of invalid observations, while invalid samples on “Assembled Payload Size” and “Operating System” where masked and removed. 
 
## 5. RANDOM FOREST 
Part of the ensemble trees family of machine learning techniques, random forests consist of creating an ensemble of tree models through bootstrapping with or without replacement of the samples to feed multiple decision trees, where the classification result that forms a majority among the k-tree predictions is selected, reducing variance. Random Forest is an improvement on the bagging method as it limits the number of features each tree can use to split the data to reduce the tree correlation. They are not as inclined to overfitting as conventional classification trees, at the expense of interpretability.
 
### 5.1 Hyperparameter Tunning
When aiming to achieve the best performance with Random Forrest models, it is crucial to consider the Out of Bag (OOB) misclassification rate. On the training process, part of the data is left “out of bag”, or not used on the training phase, so it can be used as a validation test, where the proportion of the data to work as the validation test is defined by the variable “sample.fraction”. The model then calculates the proportion of this data that was incorrectly classified, returning an OBB misclassification rate. 
 
The parameters that most influence performance results are “num.trees”, that defines the total number of trees in the Random Forrest, and “mtry” that defines the number of random features that will be used at each split. Further tunning can also be applied to the model, such as the proper “sample.fraction” divisions, “min.node.size” that defines the minimum size of the split nodes where each tree will stop further splitting the nodes, and the “replace” function can be set to TRUE or FALSE for sample bootstrapping.
 
### 5.2 Unbalanced Training Set
For the Random Forest model trained on the unbalanced training set, a search grid was set with 3 variations of “num.trees” (200,350,500),  3 variations of “mtry”(2,6,12),  5 variations of “min.node.size”(2, 4, 6, 8, 10), and both options of “replace”(TRUE or FALSE). This search grid generated 270 unique combinations where the model performance was measured on OBB misclassification on the training set and sensitivity, specificity and accuracy on the test set. The top 10 results in terms of OBB classification are shown on the table below:

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20RF%20obb%20misc%20unB.png)

The second combination on the table was chosen as the optimal parameter setting for the model with 200 trees, 6 random features, minimum node size of 4, sample fraction of 0.75, and replace function set to false, where it will not perform the sample bootstrap replacement, but the “pasting” method. It achieved a very low OBB misclassification of 0.065% and high accuracy of 99.94%. It was chosen over the first combination (with very similar performance) as it sets the “min.node.size” to 4, generating a lower complexity model than the first option would with “min.node.size” of 2. 
 
Configured with the chosen top performing parameter combination, the model achieved a very low rate of false positives with 0.02%, but a not as impressive false negative rate of 5.87%. These and other performance metrics are further discussed on topic 7, “Performance Comparison & Evaluation”. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20RF%20cm%20unB.png)

### 5.3 Balanced Training Set
The model trained on the balanced training set, had its hyperparameters tunning based on the same search grid of 270 unique combinations. The top 10 results are shown on the table below, also in terms of OBB classification.

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20RF%20obb%20misc%20Ba.png)

In this case further investigation was required. All the top 10 results obtained from the search demonstrated a very low OBB misclassification rate and almost identical specificity and accuracy numbers, but despite showing high performance on these metrics, the sensitivity rate was alarming. In fact, all the 270 combinations presented low OBB misclassification(highest value was 0.040%) and high accuracy. For the specific problem in hand, the rate of false negatives, what drives the sensitivity down is very important. As the company would be at risk of breaches if the model fails to detect malicious events, it was decided to overrule the OBB recommendation and select the combination with the highest sensitivity and consequently, the lowest false negative rate.  
 
With the combination of 200 trees, 6 random features, minimum node size of 6, sample fraction of 0.5, and replace function set to TRUE, the model achieved an also low OBB misclassification of 0.023%, what is an improvement to the model trained on unbalanced data and still high accuracy of 99.90%. Even when selecting for the lowest sensitivity, it produced a concerning false negative rate of 6.57% and false positive rate of 0.06%.

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20RF%20cm%20Ba.png)

## 6. ELASTIC-NET REGRESSION
Logistic Elastic-Net Regression is a form of penalised or regularization regression models. It can be described as a combination of the similar LASSO and Ridge techniques and has the purpose of blending the characteristics of both, in an effort to offset their limitations. The model can handle collinearity between multiple features by modifying the coefficients of each variable. It can also set to zero the coefficient of features that are less relevant to the classification. 
 
### 6.1 Hyperparameter Tunning
The balance between penalising the coefficients or pushing them to zero is determined by the “alpha” value, that ranges from 0 to 1, where the closer it is to zero the more similar to Ridge regression the model will perform, tending to penalise the coefficient instead of setting them to zero. The “lambda” value is other hyper-parameter that influenced elastic-net results, it determines the amount of penalization to be applied to each feature. Both these parameters were tuned by setting search ranges for each of them and running a k-fold “repeated” cross validation, where a portion of the training sample set for validation of the model on each combination of alpha and lambda. The “repeat” number defines how many times over, the cross validation will be run. 
 
### 6.2 Unbalanced Training Set
The Search Range was defined with 100 values between 0.001 and 1000 for lambda, and sequence of 9 values from 0.1 to 0.9 for alpha. The top performing values were 0.2 for Alpha and 0.001 for lambda, on the test ran with 5-fold cross validation and 0.1 for alpha and equal 0.001 for lambda on the 10-fold cross validation, both with 3 “repeats”. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20elaN%20cv%20plot%20unB.png)

As seen on the plot, all variations of alpha performed equally well on the lowest lambda value of 0.001. What implies that this is the parameter with the most effect on accuracy. A low lambda value determines that low levels of penalisation will be applied to the feature’s coefficients, highlighting the importance of the variance of original features. This can suggest a low level of collinearity between variables, but also a tendency to overfitting.
 
The variation with alpha value of 0.2 was selected as it achieved a slightly lower false negative rate of 12.29% than with alpha of 0.1. The other values had negligible differences across all metrics. These and other performance indicators are also discussed in more detail on topic 7. 
 
### 6.3 Balanced Training Set
The same search range with 900 unique combinations was used for optimising the parameters of the model trained on the balanced set. Again, variations of 5 and 10-fold cross validation were tested, and they produced equal results, 0.9 for alpha and 0.001 for lambda. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20elaN%20cv%20plot%20Ba.png)

Differently than on the unbalanced set, this time there were more differences in performance on each alpha variation, with the higher values prevailing over the lower ones. An 0.9 alpha value determines a model that operates more like LASSO regression. The repeated lambda value of 0.001 reinforced that the characteristics of the original data are important for classification. 
 
The lower rate of false negatives of 1.64% suggests that training the model a balanced dataset reduces its bias towards the non-malicious class, as it will not be rewarded for classifying samples as non-malicious to minimise error. The rate of false positives of 0.64% was also low, but a poorer perform than the previous models. 
 
## 7. PERFORMANCE COMPARISON & EVALUATION
When evaluating machine learning models it I crucial to understand the problem it is meant to solve and the data used for training and testing. Different metrics will be more important to different applications. For this case of predicting malicious and non-malicious connections, false positive and negative rates, accuracy, precision, recall and f-score were used to evaluate the performance of the 04 models. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20perf%20eval.png)

The false negative rate was the most concerning metric, with 03 of the models above 5.0% and on the balanced Elastic-Net below the mark with 1.64%. The false negative rate indicates how much of the malicious observations were incorrectly identified as non-malicious. All 4 models also performed well at not misclassifying non-malicious samples as malicious, with the unbalanced Elastic-Net achieving the lowest false positive rate of 0.01% and the models trained on the balanced set tending to perform slightly worse. All models presented high levels of overall accuracy, with the Random Forrest trained on the unbalanced set achieving the highest rate of 99.95%. 
 
The unbalanced Elastic-Net also presented the highest precision score of 98.52%, with its balanced counterpart performing the poorest at 47.34%. This difference reflects the balance between malicious and non-malicious observations in each training set, as precision represents the proportion of total positive predictions that were correct. The Elastic-Net model, trained this time trained on the balanced data set has the highest recall score of 98.36%, reflecting the lowest false negative rate of 1.64%. Recall is also known as sensitivity measures the proportion of actual positives that were identified correctly. F-score represents the mean of precision and recall, it is used for displaying a more overall performance of each model. The unbalanced Random Forrest was the best performing on this indicator.
 
## 8. CONCLUSION
The problem description on this project highlights the high impact of an eventual successful attack and therefore, the importance of identifying malicious events, therefore, a higher weight must be given to false negative rates and recall. It also points to the importance of the false positive rates, as dealing with too many non-malicious observations being incorrectly classified as malicious can cause alert-fatigue on the security team. 
 
The Random Forest model achieved great accuracy and F-score, but it would make for a better fit to a different problem, as having a false negative rate of 5.87% it would allow for too many malicious events to pass unflagged. For the reasons mentioned above, the Logistic Elastic-Net Regression trained on the Balanced dataset is the more suitable choice for this project’s purpose. It was able to flag the most actual malicious events with a still high overall accuracy of 99.36%. Besides scoring the lowest in precision and having the highest false positive rate of 0.64%, the differences to the other models in false negatives (4.23% to second best) is greater than its handicap on false positives (0.63% to top scorer). 






