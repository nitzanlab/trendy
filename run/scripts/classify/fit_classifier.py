import numpy as np
import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def load_features_labels(features_path, labels_dir):
    # Load features
    features = np.load(features_path)
    n_samples = features.shape[0]
    
    # Load labels and ensure correct ordering
    labels = []
    for i in range(n_samples):
        label_file = os.path.join(labels_dir, f'k_{i}.npy')
        labels.append(np.load(label_file))
    
    labels = np.array(labels).squeeze()  # Convert list of arrays to a single array and remove extra dimensions
    
    return features, labels

def main(train_features_dir, train_labels_dir, test_features_dir, test_labels_dir):
    # Load training data
    X_train, y_train = load_features_labels(train_features_dir, train_labels_dir)
    # Load testing data
    X_test, y_test = load_features_labels(test_features_dir, test_labels_dir)
    
    # Train logistic regression model
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Test the model
    y_pred = clf.predict(X_test)
    
    # Save results
    results = classification_report(y_test, y_pred, output_dict=True)
    np.save('classifier_results.npy', results)
    
    # Print results
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a logistic regression model.')
    parser.add_argument('--train_features_path', type=str, required=True, help='Path for training features')
    parser.add_argument('--train_labels_dir', type=str, required=True, help='Directory for training labels')
    parser.add_argument('--test_features_path', type=str, required=True, help='Path for testing features')
    parser.add_argument('--test_labels_dir', type=str, required=True, help='Directory for testing labels')
    
    args = parser.parse_args()
    
    main(args.train_features_path, args.train_labels_dir, args.test_features_path, args.test_labels_dir)

