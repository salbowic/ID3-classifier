import numpy as np

class Id3TreeClassifier:
    def __init__(self, max_depth: int = 20):
        """
        Initializes the ID3 decision tree classifier.

        @param max_depth: Maximum depth of the decision tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, data: np.ndarray) -> float:
        """
        Calculates the entropy of a dataset.

        @param data: Input dataset.
        @return: Entropy value of the dataset.
        """
        num_of_classes = self.count_occurrences(data[:, -1])
        probabilities = np.array(list(num_of_classes.values())) / len(data)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # +1e-10 to avoid log(0)

    def information_gain(self, data: np.ndarray, attribute: int) -> float:
        """
        Calculates information gain for a specific feature.

        @param data: Input dataset.
        @param attribute: Index of the given attribute.
        @return: Information gain for the given attribute.
        """
        attribute_values = data[:, attribute]
        values, counts = np.unique(attribute_values, return_counts=True)

        # Formula for entropy of the dataset divided into subsets by the given attribute
        subsets_entropy = np.sum(
            [(counts[i] / np.sum(counts)) * self.entropy(data[attribute_values == v])
             for i, v in enumerate(values)]
        )
        return self.entropy(data) - subsets_entropy
    
    def build_tree(self, data: np.ndarray, depth: int):
        """
        Function for recursively build the decision tree.

        @param data: Input dataset.
        @param depth: Current depth of the tree.
        @return: Decision tree node or dominant class.
        """
        unique_classes = np.unique(data[:, -1])
        if len(unique_classes) == 1:
            return unique_classes[0]
        
        if depth == 0:
            return self.find_majority_class(data[:, -1])

        # Identify the attribute with the highest information gain
        num_attributes = data.shape[1] - 1  # -1 because we subtract the class column
        inf_gains = [self.information_gain(data, attribute) for attribute in range(num_attributes)]
        best_attribute = np.argmax(inf_gains)

        # Select the root of the tree as the attribute with the highest information gain
        # and check what values it takes
        node = {best_attribute: {}}
        values = np.unique(data[:, best_attribute])

        # For each unique value of the attribute with the highest information gain
        # create a data subset that contains only those examples where the selected attribute has the given value.
        for value in values:                                                        
            subset = data[data[:, best_attribute] == value]
            if len(subset) == 0:
                node[best_attribute][value] = self.find_majority_class(data[:, -1])
            else:
                node[best_attribute][value] = self.build_tree(subset, depth - 1)

        return node

    def fit(self, X, y):
        """
        Fits the decision tree to training data.

        @param X: Feature matrix.
        @param y: Target class labels.
        """
        data = np.column_stack((X, y))
        self.tree = self.build_tree(data, self.max_depth)

    def classify(self, instance, node):
        """
        Classifies an example using the decision tree.

        @param instance: Example to be classified.
        @param tree: Current decision tree node.
        @return: Predicted class for the example.
        """
        if isinstance(node, dict):
            attribute = list(node.keys())[0]
            value = instance[attribute]

            if value in node[attribute]:
                subtree = node[attribute][value]
                return self.classify(instance, subtree)
            else:
                if isinstance(list(node[attribute].values())[0], dict):
                    return self.classify(instance, list(node[attribute].values())[0])
                else:
                    return list(node[attribute].values())[0]
        else:
            return node

    def predict(self, X):
        """
        Predicts class labels for a set of examples.

        @param X: Feature matrix.
        @return: Predicted class labels.
        """
        predictions = [self.classify(instance, self.tree) for instance in X]
        return np.array(predictions)
    
    def count_occurrences(self, data_column: np.ndarray) -> dict:
        """
        Counts occurrences of unique values in a data column.

        @param data_column: Input data column.
        @return counts: Dictionary containing counts of unique values.
        """
        counts = {}
        for value in data_column:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1
        return counts
    
    def find_majority_class(self, data_column: np.ndarray) -> str:
        """
        Finds the class that occurs most frequently in a data column.

        @param data_column: Input data column.
        @return majority_class: Class occurring most frequently in the data column.
        """
        counts = self.count_occurrences(data_column)
        majority_class = max(counts, key=counts.get)
        return majority_class
