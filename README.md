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

Model                    AccuracyTrain  AccuracyVal  AccuracyTest
0 LogisticRegression     0.850174     0.852881      0.839917
1    RandomForest        0.854255     0.844699      0.832602


We can see that Logistic is a little better so i will use other metrics on it based on validation data.

Precision: 0.7454786060873401 
Recall: 0.46326754385964913 
F1: 0.5714285714285714 
ROC-AUC: 0.8720747385147327

Now create a confusion matrix(ConfusionMatrix.png)

We can see that actually u can use this model to predict rainy days.
However, the Recall is only 0.4633, which means that the model detects only about 46% of the actual rainy days.
So i wouldn't recommend u to blind trust when model says "it won't rain tomorrow".