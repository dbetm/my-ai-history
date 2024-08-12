
def probability_of_disease(accuracy, prevalence):
    """We need to compute the probability of being tested positive and being sick and
    the probability of being tested negatively and being healthy.

    Sensitivity (recall) = TP / (TP + FP), how good for positive cases?
    Specificity = TN / (TN + FN), how good for negative cases?

    We will use the Bayes' Theorem.

    prob_sick_given_positive =>
    P(d=1, pos) = P(d=1) * P(pos|d=1) / P(pos)

    P(d=1) = 0.03 (which is the prevalence)
    P(pos|d=1) = 0.95 (which is the sensitivity)

    P(pos) = prev*acc + (1-prev)*acc
    P(pos) = 0.03*0.95 + 0.97*0.95

    We will use a similar logic to compute the probability of being tested negatively and being healthy.
    """
    sensitivity = accuracy
    specificity = accuracy
    healthy_perc = 1 - prevalence

    prob_sick_given_positive = (
        (prevalence * sensitivity) / ((accuracy * prevalence) + (healthy_perc * (1-accuracy)))
    )

    """
    X = not having disease
    Y = tested negatively

    P(d=0) = 1 - prevalence
    P(Y) = specificity

    P(d=0 | X) = P(d=0) * P(Y | d=0) / P(Y)
    """
    prob_health_given_negative = (
        ((1-prevalence) * specificity) / (((1-prevalence) * accuracy) + (prevalence * (1-accuracy)))
    )

    return [prob_sick_given_positive * 100, prob_health_given_negative * 100]



if __name__ == "__main__":
    acc = float(input())
    prevalence = float(input())

    print(probability_of_disease(acc, prevalence))
