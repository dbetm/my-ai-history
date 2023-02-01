from functools import partial

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


RANDOM_STATE = 42


# Dataset: Breast cancer - original one from: https://archive.ics.uci.edu/ml/datasets/breast+cancer

def eval_model(x_train, y_train, x_test, y_test, model):
    print("Model", type(model))
    model.fit(x_train, y_train)

    # measure the performance on test set
    y_pred = model.predict(x_test)

    print("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy", acc)


def extract_transform_and_load_data(dataset_path: str):
    dataset = pd.read_csv(dataset_path)

    print("dataset size", dataset.shape[0])

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    dataset_path = "../../../../datasets/ml_az_course/007_breast_cancer.csv"

    x_train, x_test, y_train, y_test = extract_transform_and_load_data(dataset_path)

    # create models
    logistic_regressor_model = LogisticRegression(random_state=RANDOM_STATE)
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    svc_linear_model = SVC(kernel="linear", random_state=RANDOM_STATE)
    svc_kernel_model = SVC(kernel="rbf", random_state=RANDOM_STATE)
    naive_bayes_model = GaussianNB()
    decision_tree_model = DecisionTreeClassifier(
        criterion="entropy", random_state=RANDOM_STATE
    )
    random_forest_model = RandomForestClassifier(
        n_estimators=15, criterion="entropy", random_state=42
    )

    models = [
        logistic_regressor_model,
        knn_model,
        svc_linear_model,
        svc_kernel_model,
        naive_bayes_model,
        decision_tree_model,
        random_forest_model,
    ]

    eval_ = partial(eval_model, x_train, y_train, x_test, y_test)

    # create partial
    for model in models:
        eval_(model)

        print("-"*42)


# The winner was sklearn.naive_bayes.GaussianNB with 0.9649 of accuracy score.
