import numpy as np

class Id3TreeClassifier:
    def __init__(self, max_depth: int = 20):
        """
        Inicjalizacja klasyfikatora drzewa decyzyjnego ID3.

        @param max_depth: Maksymalna głębokość drzewa decyzyjnego.
        """
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, data: np.ndarray) -> float:
        """
        Funkcja, która oblicza entropię zbioru danych.

        @param data: Zbiór danych wejściowych.
        @return: Wartość entropii zbioru danych.
        """
        num_of_classes = self.count_occurrences(data[:, -1])
        probabilities = np.array(list(num_of_classes.values())) / len(data)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10)) # +1e-10, aby uniknąć log(0)

    def information_gain(self, data: np.ndarray, attribute: str) -> float:
        """
        Oblicza zysk informacyjny dla konkretnej cechy.

        @param data: Zbiór danych wejściowych.
        @param attribute: Indeks danego atrybutu.
        @return: Zysk informacyjny dla danego atrybutu.
        """
        attribute_values = data[:, attribute]
        values, counts = np.unique(attribute_values, return_counts=True)

        # Wzór na entropię zbioru podzielonego na podzbiory przez dany atrybut
        subsets_entropy = np.sum(
            [(counts[i] / np.sum(counts)) * self.entropy(data[attribute_values == v])
            for i, v in enumerate(values)]
        )
        return self.entropy(data) - subsets_entropy
    
    def build_tree(self, data: np.ndarray, depth: int):
        """
        Funkcja do rekurencyjnego budowania drzewa decyzyjnego.

        @param data: Zbiór danych wejściowych.
        @param depth: Aktualna głębokość drzewa.
        @return: Węzeł drzewa decyzyjnego lub klasa dominująca.
        """

        unique_classes = np.unique(data[:, -1])
        if len(unique_classes) == 1:
            return unique_classes[0]
        
        if depth == 0:
            return self.find_majority_class(data[:, -1])

        # Zidentyfikowanie atrybutu o największym zysku informacyjnym
        # d = arg max_(d∈D) InfGain(d, U) 
        num_attributes = data.shape[1] - 1 # - 1 bo odejmujemy kolumnę z klasami
        inf_gains = [self.information_gain(data, attribute) for attribute in range(num_attributes)]
        best_attribute = np.argmax(inf_gains)

        # Wybieramy korzeń drzewa jako atrybut o największym zysku informacyjnym
        # i sprawdzamy jakie wartości przyjmuje
        node = {best_attribute: {}}
        values = np.unique(data[:, best_attribute])

        # Dla każdej unikalnej wartości atrybutu o największym zysku 
        # informacyjnym tworzymy podzbiór danych, który zawiera tylko te 
        # przykłady, gdzie wybrany atrybut ma daną wartość.
        # Jeżeli podzbiór jest pusty, 
        # przypisujemy do węzła klasę dominującą w całym zbiorze danych. 
        # Jeżeli podzbiór nie jest pusty, 
        # rekurencyjnie wywołujemy funkcję build_tree dla tego podzbioru.
        for value in values:                                                        
            subset = data[data[:, best_attribute] == value]
            if len(subset) == 0:
                node[best_attribute][value] = self.find_majority_class(data[:, -1])
            else:
                node[best_attribute][value] = self.build_tree(subset, depth - 1)

        return node

    def fit(self, X, y):
        """
        Funkcja, która dopasowuje drzewo decyzyjne do danych treningowych.

        @param X: Macierz atrybutów.
        @param y: Etykiety docelowe klas.
        """
        data = np.column_stack((X, y))
        self.tree = self.build_tree(data, self.max_depth)

    def classify(self, instance, node):
        """
        Funkcja, która klasyfikuje przykład za pomocą drzewa decyzyjnego.

        @param instance: Przykład do sklasyfikowania.
        @param tree: Aktualny węzeł drzewa decyzyjnego.
        @return: Przewidziana klasa dla przykładu.
        """
        # Sprawdź, czy bieżący węzeł to słownik (węzeł wewnętrzny)
        if isinstance(node, dict):
            # Pobierz atrybut i jego wartość dla bieżącego węzła
            attribute = list(node.keys())[0]
            value = instance[attribute]

            # Jeżeli wartość atrybutu znajduje się w poddrzewie
            if value in node[attribute]:
                # Rekurencyjnie klasyfikuj, używając poddrzewa
                subtree = node[attribute][value]
                return self.classify(instance, subtree)
            else:
                # Jeżeli wartość atrybutu nie znajduje się w poddrzewie, sprawdź, czy to liść
                if isinstance(list(node[attribute].values())[0], dict):
                    # Jeżeli to poddrzewo, rekurencyjnie klasyfikuj, używając klasy dominującej
                    return self.classify(instance, list(node[attribute].values())[0])
                else:
                    # Jeżeli jest liściem, zwróć klasę liścia
                    return list(node[attribute].values())[0]
        else:
            # Jeżeli jest liściem, zwróć klasę liścia
            return node

    def predict(self, X):
        """
        Przewiduje etykiety klas dla zestawu przykładów.

        @param X: Macierz atrybutów.
        @return: Przewidziane etykiety klas.
        """
        predictions = [self.classify(instance, self.tree) for instance in X]
        return np.array(predictions)
    
    def count_occurrences(self, data_column: np.ndarray) -> dict:
        """
        Funkcja, która zlicza wystąpienia unikalnych wartości w kolumnie danych.

        @param data_column: Kolumna danych wejściowych.
        @return counts: Słownik zawierający zliczenia unikalnych wartości.
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
        Funkcja znajdująca klasę, która występuje najczęściej w kolumnie danych.

        @param data_column: Kolumna danych wejściowych.
        @return majority_class: Klasa występująca najczęściej w kolumnie danych.
        """
        counts = self.count_occurrences(data_column)
        majority_class = max(counts, key=counts.get)
        return majority_class
    