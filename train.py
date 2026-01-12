import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import mlflow

data = pd.read_csv("data.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    lr = ElasticNet(alpha=0.5, l1_ratio=0.5)
    lr.fit(X_train, y_train)

    predictions = lr.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"ElasticNet model (alpha=0.5, l1_ratio=0.5):")
    print(f"  MSE: {mse}")
    print(f"  R2: {r2}")
    
    # Create a metrics file for our CI/CD pipeline to read later
    with open("metrics.txt", "w") as outfile:
        outfile.write(f"Mean Squared Error: {mse}\n")
        outfile.write(f"R2 Score: {r2}\n")