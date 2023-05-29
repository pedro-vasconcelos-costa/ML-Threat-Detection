# ML-MaliciousConnections-Classification
Classification of network connection events with Random Forest and Elastic-Net Regression in R and Python 

## 1. INTRODUCTION
This project aims to develop an early warning system for a cyber-security department to detect potential network attacks. Leveraging machine learning models, the goal is to classify real-time network events as either malicious or non-malicious. The challenge lies in the rarity of confirmed malicious events, accounting for less than 1% of all logged network events, and the need to minimize both false positives and negatives due to their significant implications. Two machine learning algorithms, Random Forest and Elastic-Net Regression will be tested and evaluated to determine the most accurate incident detection model.

## 2. BUSINESS USE 
This project directly addresses a critical issue faced by Cyber Security departments: the detection of potential network attacks in real-time. By enhancing existing Security Information and Event Management (SIEM) system with machine learning models, we aim to provide immediate classification of network events as malicious or non-malicious. This real-time feedback can significantly reduce the response time to potential threats, thereby enhancing cybersecurity defences.
 
The models developed in this project have broad applications in various sectors. In finance, similar models can be used for fraud detection, identifying suspicious transactions in real-time to prevent financial losses. In healthcare, these models can be used to detect anomalies in patient data, potentially identifying health issues before they become critical. In the field of IoT, such models can help secure devices and networks by identifying unusual patterns in data traffic. In essence, any industry dealing with large volumes of data and requiring real-time anomaly detection can benefit from the application of these machine learning models.

## 3. THE DATA 
The dataset used for this project contains 502159 observations of network connection events. It was decided to drop the IPV6.Traffic feature for having too many invalid values, so the dataset was left with 13 variables: a “Class” variable that determines if the event was confirmed as malicious or non-malicious, 3 categorical and 9 continuous features, including Operating System(source), Ingress Router, Packet Size and more. 
 
To try different training settings for each model, an UNBALANCED training set with 19800 non-malicious and 200 malicious samples, and a BALANCED training set with 19800 non-malicious and malicious samples each(malicious samples were bootstrapped to reach the desired quantity) were defined. The test set is composed by the remaining 472036 observations not used on the training sets. It is important to note that on the test set, only 2741 or 0.038% of the samples correspond to malicious events, demonstrating a great imbalance in the data. 

![image](https://github.com/pedro-vasconcelos-costa/ML-MaliciousConnections-Classification/blob/main/img_%20the%20data.png)
