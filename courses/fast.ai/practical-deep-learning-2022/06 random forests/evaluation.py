import math


def r_mse(pred, y):
    return round(math.sqrt(((pred - y)**2).mean()), 6)

def m_rmse(model, xs, y):
    return r_mse(model.predict(xs), y)