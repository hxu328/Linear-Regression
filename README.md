# Linear-Regression
Percentage of body fat, age, weight, height, and ten body circumference measurements (e.g., abdomen) are recorded for 252 men. Body fat, one measure of health, has been accurately estimated by an underwater weighing technique. Fitting body fat to the other measurements using multiple regression provides a convenient way of estimating body fat for men using only a scale and a measuring tape. In this assignment, you will be looking at the bodyfat dataset (Links to an external site: http://jse.amstat.org/v4n1/datasets.johnson.html) and build several models on top of it.

## Program Specification
You will be using the bodyfat dataset (bodyfat.csv) for this project.

1. get_dataset(filename) — takes a filename and returns the data as described below in an n-by-(m+1) array
2. print_stats(dataset, col) — takes the dataset as produced by the previous function and prints several statistics about a column of the dataset; does not return anything
3. regression(dataset, cols, betas) — calculates and returns the mean squared error on the dataset given fixed betas
4. gradient_descent(dataset, cols, betas) — performs a single step of gradient descent on the MSE and returns the derivative values as an 1D array
5. iterate_gradient(dataset, cols, betas, T, eta) — performs T iterations of gradient descent starting at the given betas and prints the results; does not return anything
6. compute_betas(dataset, cols) — using the closed-form solution, calculates and returns the values of betas and the corresponding MSE as a tuple
7. predict(dataset, cols, features) — using the closed-form solution betas, return the predicted body fat percentage of the give features.
8. synthetic_datasets(betas, alphas, X, sigma) — generates two synthetic datasets, one using a linear model and the other using a quadratic model.
9. plot_mse() — fits the synthetic datasets, and plots a figure depicting the MSEs under different situations.
