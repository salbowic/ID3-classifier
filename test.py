import pandas as pd
import numpy as np
from id3_classifier import *
from ucimlrepo import fetch_ucirepo 
from sklearn.metrics import accuracy_score, confusion_matrix

def shuffle_and_split_data(X, Y, ratio, random_seed = 100):
    np.random.seed(random_seed)

    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    Y_shuffled = Y[shuffled_indices]

    split_index = int(len(X) * ratio)

    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    Y_train, Y_test = Y_shuffled[:split_index], Y_shuffled[split_index:]

    return X_train, X_test, Y_train, Y_test

def prepare_data(data_id, ratio, random_seed = 100):
    dataset = fetch_ucirepo(id = data_id)
    X = dataset.data.features.values.astype(str)
    Y = dataset.data.targets.values.astype(str)
    
    X_train, X_test, Y_train, Y_test = shuffle_and_split_data(X, Y , ratio, random_seed)

    return X_train, X_test, Y_train, Y_test

def cut_and_prepare_data(data_id, ratio, percentage, random_seed = 100):
    X_train, X_test, Y_train, Y_test = prepare_data(data_id, ratio, random_seed)
    p = percentage/100
    lengths = [int(p * len(X_train)), int(p * len(X_test)), int(p * len(Y_train)), int(p * len(Y_test))]
    X_train, X_test = X_train[:lengths[0]], X_test[:lengths[1]]
    Y_train, Y_test = Y_train[:lengths[2]], Y_test[:lengths[3]]
    return X_train, X_test, Y_train, Y_test

def accuracy_test(data_id, ratio, random_seed = 100):
    X_train, X_test, Y_train, Y_test = cut_and_prepare_data(data_id, ratio, 3.5, random_seed)
    print(".", end='', flush=True) # to show progress
    id3_classifier = Id3TreeClassifier()
    id3_classifier.fit(X_train, Y_train)

    predictions = id3_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)
    cm = confusion_matrix(Y_test, predictions)

    unique_classes = np.unique(Y_test)

    cm_df = pd.DataFrame(cm, index=[f"{label} (real)" for label in unique_classes], 
                         columns=[f"{label} (predicted)" for label in unique_classes])

    return accuracy, cm_df

def test_dif_random_seeds(id):
    total_accuracy1 = 0
    cm1_shape = accuracy_test(id, 0.6, 1)[1].shape
    average_cm1 = np.zeros(cm1_shape)

    for random_seed in range (1,101):
        accuracy1, cm1 = accuracy_test(id, 0.6, random_seed)
        total_accuracy1 += accuracy1
        average_cm1 += cm1

    average_accuracy1 = total_accuracy1 / 100 
    average_cm1 /= 100

    print("Average Accuracy 1:", average_accuracy1)
    print("Average Confusion Matrix 1:")
    print(average_cm1)

def test_dif_datasets_with_categorical_feature_type(ids):
    for id in ids:
        accuracy, cm = accuracy_test(id, 0.6, 42)
        print(f"Accuracy (id = {id}): {accuracy}")
        print(cm)

    print(cm)

def main():
    # ids = [14, 73, 19, 105]
    # test_dif_datasets_with_categorical_feature_type(ids)
    # for id in ids[:2]:
    #     test_dif_random_seeds(id)
    
    test_dif_random_seeds(73)

    # X_train, X_test, Y_train, Y_test = cut_and_prepare_data(data_id=14, ratio=0.6, percentage=10, random_seed=42)
    # print(X_train, X_test, Y_train, Y_test)
        
if __name__ == "__main__":
    main()