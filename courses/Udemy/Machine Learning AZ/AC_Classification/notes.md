## Logistic regression

Predict a categorical dependendent varoable from a number of indepedent variables.

`ln( p / 1-p ) = b_0 + b_1 * X_1`

**Intuition**

Predict (ye / no) if a person will purchae health insurance.

![](./assets/logistic_regr_intuition.png)

Notice that the curve is a sigmoide function.



We can set a threshold for probabilities in order to decide a category.

The likelihood of a curve is simply multiply the outcome probabilities of some set of data points, then, we will be selecting the best S-curve to be that with the **Maximum Likelikhood**.