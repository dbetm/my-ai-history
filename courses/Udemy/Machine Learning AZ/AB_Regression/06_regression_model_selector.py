import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


class DataLoader:
    def __init__(self, local_path: str) -> None:
        dataset = pd.read_csv(local_path)
        self.x = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

    def reshape_target(self):
        self.y = self.y.reshape(len(self.y), 1)
    

    def get_splitted_data(self, test_size: float = 0.2):
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=42
        )

        return x_train, x_test, y_train, y_test


def evaluate(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

    print("\nr2 score", r2_score(y_test, y_pred))


if __name__ == "__main__":
    dataset_path = "../../../../datasets/ml_az_course/005_combined_cycle_power_plant.csv"

    data_loader = DataLoader(local_path=dataset_path)
    x_train, x_test, y_train, y_test = data_loader.get_splitted_data()

    # multiple linear regression
    print("---- Multiple linear regression ----")
    mult_linear_regressor = LinearRegression()
    mult_linear_regressor.fit(X=x_train, y=y_train)

    y_pred = mult_linear_regressor.predict(x_test)
    evaluate(y_test, y_pred)

    # polynomial regression
    print("\n----- Polynomial regression -----")
    polynomial_regressor = PolynomialFeatures(degree=4)
    x_poly = polynomial_regressor.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(X=x_poly, y=y_train)

    y_pred = regressor.predict(polynomial_regressor.transform(x_test))
    evaluate(y_test, y_pred)

    # decision tree regression
    print("\n--- Decision Tree Regression ---")
    tree_regressor = DecisionTreeRegressor(random_state=42)
    tree_regressor.fit(X=x_train, y=y_train)

    y_pred = tree_regressor.predict(x_test)
    evaluate(y_test, y_pred)

    # random forest regression
    print("\n --- Random forest regression ----")
    random_forest_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    random_forest_regressor.fit(X=x_train, y=y_train)

    y_pred = random_forest_regressor.predict(X=x_test)
    evaluate(y_test, y_pred)

    # suport vector regression
    print("\n--- Support Vector Regression ---")

    data_loader = DataLoader(local_path=dataset_path)
    data_loader.reshape_target()
    x_train, x_test, y_train, y_test = data_loader.get_splitted_data()

    sc_x = StandardScaler()
    sc_y = StandardScaler()

    x_train = sc_x.fit_transform(x_train)
    y_train = sc_y.fit_transform(y_train)

    regressor = SVR(kernel="rbf")
    regressor.fit(X=x_train, y=y_train)

    y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(x_test)).reshape(-1, 1))
    evaluate(y_test, y_pred)


