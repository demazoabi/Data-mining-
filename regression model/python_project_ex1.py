import numpy as np
import pandas as pd

# Filename variables
python_file="python_project_ex1.py"
jupyter_file="LinearRegression"
PDF_file="model_conclusions"


def rmse_cal(y_test, y_pred):
    squared_errors = (y_test - y_pred) ** 2
    sse = np.sum(squared_errors)
    mse = sse / len(y_test)
    rmse = np.sqrt(mse)
    return rmse


def score_linear(x, beta):
    y_pred = np.dot(x, beta)
    return y_pred


def loss(y, score_y):
    squared_errors = (y - score_y) ** 2
    sse = np.sum(squared_errors)
    return sse


def grad_loss(x, y, beta):
    y_pred = np.dot(x, beta)
    grad = -2 * np.dot(x.T, (y - y_pred))
    return grad


def gradient_descent_step(beta, grad_beta, learning_rate):
    beta_new = beta - learning_rate * grad_beta
    return beta_new


def main_func (file_path, target,columns_to_encode,learning_rate=0.01,iterations=10):
    data = pd.read_csv(file_path)
    data=pd.get_dummies(data, columns=data.columns[columns_to_encode], drop_first=True)
    y = data[target]
    x = data.drop(target, axis=1)
    beta = np.zeros(x.shape[1])
    for iteration in range(iterations):
        grad = grad_loss(x, y, beta)
        beta = gradient_descent_step(beta, grad, learning_rate)
        y_pred = np.dot(x, beta)
        curr_loss = loss(y, y_pred)
        print(f"Iteration {iteration + 1}: Beta = {beta}, Loss = {curr_loss}")

    return beta


main_func("C:\\Housing.csv","furnishingstatus_unfurnished",[5,6,7,8,9,11,12])

