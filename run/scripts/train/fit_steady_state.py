import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import product

# Generate polynomial features up to the given degree for 2 variables
def poly_features(X, degree):
    n_samples, n_vars = X.shape
    powers = [p for p in product(range(degree + 1), repeat=n_vars) if sum(p) <= degree]
    features = np.empty((n_samples, len(powers)))
    for i, p in enumerate(powers):
        features[:, i] = np.prod(X ** np.array(p), axis=1)
    return features, powers

# Fit a rational function to the data
def fit_rational_function(X, Y, degree_num, degree_den):
    # Generate polynomial features for the numerator and denominator
    X_poly_num, _ = poly_features(X, degree_num)
    X_poly_den, _ = poly_features(X, degree_den)

    # Set up the system for least squares: P(X) / Q(X)
    A = X_poly_num / (X_poly_den[:, [0]] + 1e-6)  # Corresponds to f1(x, y)
    #A2 = X_poly_num / (X_poly_den[:, [0]] + 1e-6)  # Corresponds to f2(x, y)

    # Fit the model

    all_coeffs = []
    for i in range(Y.shape[-1]):
        coeffs = np.linalg.lstsq(A, Y[:, i], rcond=None)[0]
        all_coeffs.append(coeffs)
    #coeffs_p2 = np.linalg.lstsq(A2, Y[:, 1], rcond=None)[0]

    return all_coeffs, degree_num, degree_den

# Simplified evaluation function
def eval_rational_function(X, all_coeffs, degree_num, degree_den):
    # Generate polynomial features for the test set using the same degree as for the training set
    X_poly_num, _ = poly_features(X, degree_num)
    X_poly_den, _ = poly_features(X, degree_den)

    # Make predictions by evaluating the rational function
    all_preds = []
    for i, coeffs in enumerate(all_coeffs):
        pred = (X_poly_num @ coeffs) / (X_poly_den[:, 0] + 1e-6)
        all_preds.append(pred)

    return np.column_stack(all_preds)

# Create a minimal working example with different-sized train and test sets
def main():
    # Generate toy data for X_train and Y_train
    num_dims = 4

    # Parameters
    X_train = np.squeeze(np.load(f'./data/ss{num_dims}/all_params.npy')[:,:2])  # 100 samples, 2 features
    Y_train = np.load(f'./data/ss{num_dims}/all_ss.npy')  # 100 samples, 2 targets

    # Generate toy data for X_test with a different number of samples
    #X_test = np.random.rand(50, 2)  # 50 samples, 2 features
    col1 = np.linspace(1.5, 4.0,100)
    col2 = np.ones(100)

    X_test = np.column_stack((col1,col2))

    # Fit the rational function
    degree_numerator = 2
    degree_denominator = 2
    all_coeffs, degree_num, degree_den = fit_rational_function(X_train, Y_train, degree_numerator, degree_denominator)

    # Evaluate the model on the test set
    Y_pred = eval_rational_function(X_test, all_coeffs, degree_num, degree_den)

    np.save(f'/cs/labs/mornitzan/ricci/projects/bifurcation/run/data/ss{num_dims}/est_ss.npy', Y_pred)

    # Print predictions on the test set
    #print("Predictions on X_test:")
    #print(Y_pred)

    #fig, axes = plt.subplots(1,2, figsize=(10,5))
    #axes[0].plot(X_test[:,0], Y_pred[:,0])
    #axes[0].set_xlabel(r'$A$')
    #axes[0].set_ylabel(r'$S_{0,ss}$')
    #axes[1].plot(X_test[:,0], Y_pred[:,1])
    #axes[1].set_xlabel(r'$A$')
    #axes[1].set_ylabel(r'$S_{1,ss}$')

    #plt.savefig('./figs/steady_state.png')
    #plt.close()

if __name__ == '__main__':
    main()

