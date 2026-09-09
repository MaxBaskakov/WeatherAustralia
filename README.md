# My first ai model 

I made this project to predict probability of rain tomorrow based on dataset from Kaggle 

https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package?resource=download

I will use two classification models Logistic and RandomForest and see witch one is better to my task.

I prepared data for models  
-splitting the data into train, validation, and test sets;
-removing rows with missing values in RainTomorrow;
-filling missing values in numerical columns using the mean;
-scaling numerical features;
-encoding categorical features using One-Hot Encoding.

Then i compared 2 models based on their accuracy 

| Model | Accuracy Train | Accuracy Val | Accuracy Test |
|---|---:|---:|---:|
| Logistic Regression | 0.850174 | 0.852881 | 0.839917 |
| Random Forest | 0.854871 | 0.845453 | 0.832140 |



We can see that Logistic is a little better so i will use other metrics on it based on validation data.

**Precision: 0.7455**
**Recall: 0.4633** 
**F1: 0.57142** 
**ROC-AUC: 0.8721**

Now create a confusion matrix(ConfusionMatrix.png)

The ROC-AUC score of 0.8721 shows that the model is quite good at distinguishing between rainy and non-rainy days.
However, the Recall is only 0.4633, which means that the model detects only about 46% of the actual rainy days.
So i wouldn't recommend u to blind trust when model says "it won't rain tomorrow".