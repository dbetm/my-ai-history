import re
from functools import partial
from typing import Any, Tuple

import pandas as pd
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


RANDOM_STATE = 42


def eval_model(x_train, y_train, x_test, y_test, model: Any):
    print("Model", type(model))
    model.fit(x_train, y_train)

    # measure the performance on test set
    y_pred = model.predict(x_test)

    print("Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("-"*23)

    print(
        "acc", accuracy_score(y_test, y_pred),
        "precision", precision_score(y_test, y_pred),
        "recall", recall_score(y_test, y_pred),
        "f1_score", f1_score(y_test, y_pred)
    )


def preprocess_review(review: str, stop_words: set) -> str:
    ps = PorterStemmer()
    # replace non-characters with a space
    review = re.sub(pattern="[^a-zA-z]", repl=" ", string=review)
    # lowercase
    review = review.lower()
    # split the reviews
    review = review.split()
    # steem the words of the review
    review = [
        ps.stem(word) for word in review if word not in stop_words
    ]
    # get back the review to the original state
    return " ".join(review)


def extract_transform_and_load_data(
    dataset_path: str, sample_percent_size_words: float = 0.8
) -> Tuple:
    # quoting value 3 to ignore doble quotes in columns
    dataset = pd.read_csv(dataset_path, delimiter="\t", quoting=3)

    print("dataset size", dataset.shape)

    # Cleaning
    corpus = list()
    en_stop_words = stopwords.words("english")
    en_stop_words.remove("not") # this sopt word is important for the problem

    for _, row in dataset.iterrows():
        # append to corpus
        review = preprocess_review(row["Review"], en_stop_words)
        corpus.append(review)
    
    # There are a lot of words that are non-stop words but they don't help 
    # to predict at all if the review is positive or negative. 
    # The trick is to use only the most frequent words.
    cv = CountVectorizer()
    sparse_matrix_x = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    total_words = len(sparse_matrix_x[0])
    cv = CountVectorizer(
        max_features=int(total_words * sample_percent_size_words)
    )
    x = cv.fit_transform(corpus).toarray()

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    return (x_train, x_test, y_train, y_test, cv)


if __name__ == "__main__":
    dataset_path = "../../../../datasets/ml_az_course/011_Restaurant_Reviews.tsv"

    (
        x_train, x_test, y_train, y_test, count_vect
    ) = extract_transform_and_load_data(dataset_path)

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

    # create partial
    eval_ = partial(eval_model, x_train, y_train, x_test, y_test)

    for model in models:
        eval_(model)

        print("-"*42)


"""The winner was SVC linear

acc         0.7700
precision   0.8085
recall      0.7307
f1_score    0.7676
"""
