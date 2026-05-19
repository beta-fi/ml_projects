import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import random 

xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([5, 4, 6, 5, 6, 7])
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step          # fix 3
    xs = [i for i in range(len(ys))]
    return np.array(xs), np.array(ys)


xs, ys = create_dataset(40, 40, 2, correlation='pos')
def best_fit_slope_and_intercept(xs, ys):
    m = (np.mean(xs) * np.mean(ys) - np.mean(xs * ys)) / (np.mean(xs)**2 - np.mean(xs**2))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    ys_orig = np.array(ys_orig)
    ys_line = np.array(ys_line)
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    ys_orig = np.array(ys_orig)
    ys_line = np.array(ys_line)
    y_mean_line = [np.mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_intercept(xs, ys)  # ✅ Called only once now
regression_line = [(m * x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)  # ✅ Now actually called
print("R²:", r_squared)  # ✅ Now actually printed

predict_x = 8
predict_y = (m * predict_x) + b

plt.scatter(xs, ys, label='Data points')
plt.scatter(predict_x, predict_y, label='Prediction')
plt.plot(xs, regression_line, label='Regression line')
plt.legend()
plt.show()