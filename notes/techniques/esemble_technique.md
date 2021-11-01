# Ensemble technique

An ensemble is a collection of predictors. For example, instead of using a single model (say, **logistic regression**) for a classification problem, we can use multiple models (say, **logistic regression** + **decision trees**, etc) to perform predictions.

Ensemble methods can be implemented by either **bagging* or **boosting**.

## Bagging

Bagging is a technique where in we build independent models/predictors using a random subsample/bootstrap of data for each of the models/predictors. Then an average (weighted, normal, or by voting) of the scores from the different predictors is taken to get the final score/prediction. The most famous bagging method is **random forest**.

## Boosting

The predictors are not independently trained but done so in a sequential manner. The aim of this sequential training is for the subsequent models to learn from the mistakes of the previous model. **Gradient boosting** is an example of a boosting method.

### Gradient boosting
The main difference between gradient boosting compared to other boosting methods is that instead of incrementing the weights of misclassified outcomes from one previous learner to the next, we optimize the loss function  of the previous learner.
