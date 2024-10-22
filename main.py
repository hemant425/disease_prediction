import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('/disease_prediction_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

def predict_disease():
    print("Enter the following details for disease prediction:")
    age = float(input("Age: "))
    bmi = float(input("BMI: "))
    glucose_level = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    cholesterol = float(input("Cholesterol: "))

    user_data = pd.DataFrame([[age, bmi, glucose_level, blood_pressure, cholesterol]],
                             columns=['age', 'bmi', 'glucose_level', 'blood_pressure', 'cholesterol'])

    user_data_normalized = scaler.transform(user_data)
    prediction = knn.predict(user_data_normalized)

    if prediction == 1:
        print("\nPrediction: You may have the disease.")
    else:
        print("\nPrediction: You may not have the disease.")

if __name__ == "__main__":
    predict_disease()
