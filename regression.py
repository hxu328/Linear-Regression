import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import random



# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    points = []
    with open(filename, newline='', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            point = []
            BODYFAT = float(row['BODYFAT'])
            DENSITY = float(row['DENSITY'])
            AGE = float(row['AGE'])
            WEIGHT= float(row['WEIGHT'])
            HEIGHT= float(row['HEIGHT'])
            ADIPOSITY = float(row['ADIPOSITY'])
            NECK = float(row['NECK'])
            CHEST = float(row['CHEST'])
            ABDOMEN = float(row['ABDOMEN'])
            HIP = float(row['HIP'])
            THIGH = float(row['THIGH'])
            KNEE = float(row['KNEE'])
            ANKLE = float(row['ANKLE'])
            BICEPS = float(row['BICEPS'])
            FOREARM = float(row['FOREARM'])
            WRIST= float(row['WRIST'])
            point.append(BODYFAT)
            point.append(DENSITY)
            point.append(AGE)
            point.append(WEIGHT)
            point.append(HEIGHT)
            point.append(ADIPOSITY)
            point.append(NECK)
            point.append(CHEST)
            point.append(ABDOMEN)
            point.append(HIP)
            point.append(THIGH)
            point.append(KNEE)
            point.append(ANKLE)
            point.append(BICEPS)
            point.append(FOREARM)
            point.append(WRIST)
            points.append(point)
    dataset = np.array(points)
    return dataset


def print_stats(dataset, col):
    num_points = len(dataset)
    total = 0
    for i in range(num_points):
        total += dataset[i, col]
    mean = total / num_points
    total_sqr = 0
    for i in range(num_points):
        total_sqr += pow(dataset[i, col] - mean, 2)
    deviation = math.sqrt((1/num_points) * total_sqr)
    print(num_points)
    print(format(mean, '.2f'))
    print(format(deviation, '.2f'))


def regression(dataset, cols, betas):
    mse = 0
    for point in dataset:
        product = 0
        for i in range(len(cols)):
            product += point[cols[i]] * betas[i+1]
        product += (betas[0] - point[0])
        mse += pow(product, 2)
    mse = mse / len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    grads = []
    for i in range(len(betas)):
        sum = 0
        for point in dataset:
            product = 0
            for j in range(len(cols)):
                product += point[cols[j]] * betas[j + 1]
            product += (betas[0] - point[0])
            if i == 0:
                sum += product
            else:
                sum += product * point[cols[i-1]]
        grads.append((2 / len(dataset)) * sum)
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    init_betas = np.array(betas)
    i = 1
    while i <= T:
        init_betas = init_betas - eta * gradient_descent(dataset, cols, init_betas.tolist())
        mse = regression(dataset, cols, init_betas.tolist())
        output = []
        output.append(i)
        output.append(format(mse, '.2f'))
        for beta in init_betas.tolist():
            output.append(format(beta, '.2f'))
        print(*output)
        i += 1


def compute_betas(dataset, cols):
    x = np.ones(len(dataset))
    for i in cols:
        append_xi = dataset[:, i]
        x = np.vstack((x, append_xi))

    y = dataset[:, 0]
    y = np.array([y])
    y = y.T

    betas = np.linalg.inv(x @ x.T) @ x @ y
    betas = betas.ravel()
    betas = betas.tolist()
    mse = regression(dataset, cols, betas)
    return (mse, *betas)


def predict(dataset, cols, features):
    betas = compute_betas(dataset, cols)
    result = 0
    for i in range(len(features)):
        result += features[i] * betas[i+2]
    result += betas[1]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    betas_list = betas.tolist()
    alphas_list = alphas.tolist()
    x_list = X.ravel()
    x_list = x_list.tolist()

    linear_dataset = []
    for i in range(len(x_list)):
        product = betas_list[0] + betas_list[1] * x_list[i] + np.random.normal(0, sigma)
        linear_dataset.append([product, x_list[i]])
    linear_dataset = np.array(linear_dataset)

    quad_dataset = []
    for i in range(len(x_list)):
        product = alphas_list[0] + alphas_list[1] * pow(x_list[i], 2) + np.random.normal(0, sigma)
        quad_dataset.append([product, x_list[i]])
    quad_dataset = np.array(quad_dataset)

    return (linear_dataset, quad_dataset)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # Create an input array X
    X = []
    for i in range(1000):
        X.append(random.uniform(-100, 100))
    X = np.array(X)
    X = np.array([X])
    X = X.T

    # Create couples of betas and alphas with non-zero values
    betas = [1, 2]
    betas = np.array(betas)

    alphas = [3, 4]
    alphas = np.array(alphas)

    # Set sigmas to be: 10^{-4},10^{-3},...,1,10,...,10^5
    sigmas = []
    sigmas_init = pow(10, -4)
    for i in range(10):
        sigmas.append(sigmas_init)
        sigmas_init = sigmas_init * 10

    # Under each settings of sigmas, generate two synthetic datasets
    linears = []
    quadratics = []
    for sigma in sigmas:
        linear, quadratic = synthetic_datasets(betas, alphas, X, sigma)
        linears.append(linear)
        quadratics.append(quadratic)

    # Fit both datasets using compute_betas(), obtain the corresponding MSEs
    linear_mse = []
    quad_mse = []
    for i in linears:
        mse = compute_betas(i, cols=[1])
        linear_mse.append(mse[0])

    for i in quadratics:
        mse = compute_betas(i, cols=[1])
        quad_mse.append(mse[0])

    linear_mse = np.array(linear_mse)
    quad_mse = np.array(quad_mse)
    # plot
    plt.plot(sigmas, linear_mse, '-o', label='linear')
    plt.plot(sigmas, quad_mse, '-o', label='quadratic')
    plt.xlabel('sigma')
    plt.ylabel('MSE')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig('mse.pdf')


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()






