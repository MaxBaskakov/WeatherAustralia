import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)

sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams["figure.facecolor"] = '#00000000'

url = 'weatherAUS.csv'
raw_df = pd.read_csv(url)

raw_df.dropna(subset=["RainTomorrow"], inplace=True)
raw_df["RainTomorrow"] = raw_df["RainTomorrow"].map({"Yes": 1, "No": 0})
input_cols = list(raw_df.columns[1:-1])
target_cols = "RainTomorrow"

year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

train_inputs = train_df[input_cols].copy()
val_inputs = val_df[input_cols].copy()
test_inputs = test_df[input_cols].copy()

train_target = train_df[target_cols].copy()
val_target = val_df[target_cols].copy()
test_target = test_df[target_cols].copy()

numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
object_cols = train_inputs.select_dtypes(include="object").columns.tolist()

print(numeric_cols)
print(object_cols)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer.fit(train_inputs[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_inputs[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

encoder.fit(train_inputs[object_cols])

encoder_cols = list(encoder.get_feature_names_out(object_cols))

train_inputs[encoder_cols] = encoder.transform(train_inputs[object_cols])
val_inputs[encoder_cols] = encoder.transform(val_inputs[object_cols])
test_inputs[encoder_cols] = encoder.transform(test_inputs[object_cols])

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_inputs[numeric_cols + encoder_cols], train_target)

from sklearn.metrics import accuracy_score

train_predict = model.predict(train_inputs[numeric_cols+encoder_cols])
val_predict = model.predict(val_inputs[numeric_cols+encoder_cols])
test_predict = model.predict(test_inputs[numeric_cols+encoder_cols])

train_accuracy = accuracy_score(train_target, train_predict)
val_accuracy = accuracy_score(val_target, val_predict)
test_accuracy = accuracy_score(test_target, test_predict)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=21, n_jobs=-1)
rf_model.fit(train_inputs[numeric_cols+encoder_cols], train_target)

rf_train_predict = rf_model.predict(train_inputs[numeric_cols+encoder_cols])
rf_val_predict = rf_model.predict(val_inputs[numeric_cols+encoder_cols])
rf_test_predict = rf_model.predict(test_inputs[numeric_cols+encoder_cols])

rf_train_accuracy = accuracy_score(train_target, rf_train_predict)
rf_val_accuracy = accuracy_score(val_target, rf_val_predict)
rf_test_accuracy = accuracy_score(test_target, rf_test_predict)

print(f"Accuracy of train data on logistic model - {train_accuracy} \nAccuracy based on same data but with Random Forest - {rf_train_accuracy}")
print()
print(f"Accuracy of val data on logistic model - {val_accuracy} \nAccuracy based on same data but with Random Forest - {rf_val_accuracy}")
print()
print(f"Accuracy of test data on logistic model - {test_accuracy} \nAccuracy based on same data but with Random Forest - {rf_test_accuracy}")

results = pd.DataFrame({
    "Model": ["LogisticRegression", "RandomForest"],
    "AccuracyTrain": [train_accuracy, rf_train_accuracy],
    "AccuracyVal": [val_accuracy, rf_val_accuracy],
    "AccuracyTest": [test_accuracy, rf_test_accuracy]
})
print()
print(results)

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

prec = precision_score(val_target, val_predict)
rec = recall_score(val_target, val_predict)
f1 = f1_score(val_target, val_predict)
val_proba = model.predict_proba(val_inputs[numeric_cols+encoder_cols])
val_proba_yes = val_proba[:,1]
roc = roc_auc_score(val_target, val_proba_yes)

print()
print(f"Metrics based on validation data \nPrecision:{prec} \nRecall:{rec} \nF1:{f1} \nROC-AUC{roc}")

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(val_target, val_predict)
plt.figure(figsize=(10,8))
sns.heatmap(matrix, annot=True, fmt ="d", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix")
plt.xlabel("Model Prediction")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("ConfusionMatrix.png")


def new_data_predict_logistic(data: dict) -> str:
    input_df = pd.DataFrame([data])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoder_cols] = encoder.transform(input_df[object_cols])
    new_data_prediction = model.predict(input_df[numeric_cols+encoder_cols])
    new_data_proba = model.predict_proba(input_df[numeric_cols+encoder_cols])
    if new_data_prediction[0] == 1:
        return f"\nTomorrow will be rain with {((new_data_proba[0][1])*100):.2f}% chance"
    else:
        return f"\nTomorrow won't be rain with {((new_data_proba[0][0])*100):.2f}% chance"



example_data = {'Date': '2026-09-07',
                'Location': 'Launceston',
                'MinTemp': 23.2,
                'MaxTemp': 33.2,
                'Rainfall': 10.2,
                'Evaporation': 4.2,
                'Sunshine': np.nan,
                'WindGustDir': 'NNW',
                'WindGustSpeed': 52.0,
                'WindDir9am': 'NW',
                'WindDir3pm': 'NNE',
                'WindSpeed9am': 13.0,
                'WindSpeed3pm': 20.0,
                'Humidity9am': 89.0,
                'Humidity3pm': 58.0,
                'Pressure9am': 1004.8,
                'Pressure3pm': 1001.5,
                'Cloud9am': 8.0,
                'Cloud3pm': 5.0,
                'Temp9am': 25.7,
                'Temp3pm': 30.0,
                'RainToday': 'Yes'}

print(new_data_predict_logistic(example_data))