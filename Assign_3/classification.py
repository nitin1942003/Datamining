import math
from collections import Counter
from typing import List, Dict, Tuple, Optional

class Matrix:
    def __init__(self, data_file: str = None, data: List[List[str]] = None):
        self.matrix = []
        if data_file:
            with open(data_file, 'r') as f:
                for line in f:
                    row = line.strip().split()
                    if row:
                        self.matrix.append(row)
        elif data:
            self.matrix = data

    def element(self, i: int, j: int) -> str:
        return self.matrix[i][j]

    def size_x(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    def size_y(self) -> int:
        return len(self.matrix)

    def get_attributes(self) -> List[str]:
        return self.matrix[0][:-1] if self.matrix else []

    def get_attributes_values(self) -> Dict[str, List[str]]:
        attributes_values = {}
        for j in range(self.size_x()):
            values = [self.matrix[i][j] for i in range(1, self.size_y())]
            attributes_values[self.matrix[0][j]] = sorted(list(set(values)))
        return attributes_values

    def get_attribute_values(self, attribute: str) -> List[str]:
        return self.get_attributes_values().get(attribute, [])

    def get_score_range(self) -> List[str]:
        return self.get_attribute_values(self.matrix[0][-1])

    def attribute_index(self, attribute: str) -> int:
        return self.matrix[0].index(attribute)

    def get_attribute_values_scores(self, attribute: str) -> Dict[str, List[str]]:
        idx = self.attribute_index(attribute)
        values_scores = {}
        for value in self.get_attribute_values(attribute):
            scores = []
            for i in range(1, self.size_y()):
                if self.matrix[i][idx] == value:
                    scores.append(self.matrix[i][-1])
            values_scores[value] = scores
        return values_scores

    def get_scores(self) -> List[str]:
        return [row[-1] for row in self.matrix[1:]]

    def filter_matrix(self, attribute: str, value: str) -> 'Matrix':
        idx = self.attribute_index(attribute)
        # Keep the header row and filter the data rows
        new_matrix = [self.matrix[0][:idx] + self.matrix[0][idx+1:]]  # Keep header
        for row in self.matrix[1:]:  # Skip header
            if row[idx] == value:
                new_matrix.append(row[:idx] + row[idx+1:])
        return Matrix(data=new_matrix)

    def display(self):
        for row in self.matrix:
            print(' '.join(row))

def get_unique_scores(scores: List[str]) -> List[str]:
    return sorted(list(set(scores)))

def get_frequent_score(scores: List[str]) -> str:
    return Counter(scores).most_common(1)[0][0]

def compute_entropy(scores: List[str]) -> float:
    if not scores:
        return 0.0
    
    score_counts = Counter(scores)
    total = len(scores)
    entropy = 0.0
    
    for count in score_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    
    return entropy

def compute_attribute_entropy_gain(matrix: Matrix, attribute: str) -> float:
    scores = matrix.get_scores()
    original_entropy = compute_entropy(scores)
    
    values_scores = matrix.get_attribute_values_scores(attribute)
    after_entropy = 0.0
    
    for value_scores in values_scores.values():
        p = len(value_scores) / len(scores)
        after_entropy += p * compute_entropy(value_scores)
    
    return original_entropy - after_entropy

class Tree:
    def __init__(self):
        self.node = ""
        self.branch = ""
        self.children = []

    def build_tree(self, matrix: Matrix) -> 'Tree':
        scores = matrix.get_scores()
        unique_scores = get_unique_scores(scores)
        
        if len(unique_scores) == 1:
            self.node = unique_scores[0]
            return self
        
        if matrix.size_x() == 1:
            self.node = get_frequent_score(scores)
            return self
        
        attributes = matrix.get_attributes()
        max_gain = -1.0  # Initialize to negative value
        max_attribute = ""
        
        for attribute in attributes:
            gain = compute_attribute_entropy_gain(matrix, attribute)
            if gain > max_gain:
                max_gain = gain
                max_attribute = attribute
        
        if max_attribute == "":  # No attribute with positive gain
            self.node = get_frequent_score(scores)
            return self
            
        self.node = max_attribute
        values = matrix.get_attribute_values(max_attribute)
        
        for value in values:
            new_matrix = matrix.filter_matrix(max_attribute, value)
            child = Tree()
            child.branch = value
            
            if new_matrix.size_x() == 1 or len(new_matrix.get_scores()) == 0:
                child.node = get_frequent_score(scores)  # Use parent's scores for empty matrix
            else:
                child.build_tree(new_matrix)
            
            self.children.append(child)
        
        return self

    def print_tree(self, depth: int = -1):
        indent = "\t" * max(0, depth)
        if self.branch:
            print(f"{indent}{self.branch}")
            indent += "\t"
        if depth == -1 and self.branch:
            print("\t", end="")
        print(f"{indent}{self.node}")
        for child in self.children:
            child.print_tree(depth + 1)

    def test_tree(self, matrix: Matrix) -> List[str]:
        attributes = matrix.get_attributes()
        score_range = matrix.get_score_range()
        test_scores = []
        
        for i in range(1, matrix.size_y()):
            values = [matrix.element(i, j) for j in range(len(attributes))]
            prediction = self.temp_test_tree(attributes, values, score_range)
            test_scores.append(prediction)
        
        return test_scores

    def temp_test_tree(self, attributes: List[str], values: List[str], score_range: List[str]) -> str:
        # If current node is a leaf node (contains a score)
        if self.node in score_range:
            return self.node
        
        # Find the matching attribute
        for i, attribute in enumerate(attributes):
            if self.node == attribute:
                # Find the matching branch for this attribute value
                for child in self.children:
                    if child.branch == values[i]:
                        # If child node is a leaf node
                        if child.node in score_range:
                            return child.node
                        
                        # If not a leaf node, recursively traverse the tree
                        new_attributes = [a for j, a in enumerate(attributes) if j != i]
                        new_values = [v for j, v in enumerate(values) if j != i]
                        return child.temp_test_tree(new_attributes, new_values, score_range)
                
                # If no matching branch is found, return the most common score
                return get_frequent_score(score_range)
        
        # Default case: return the first score in the range
        return score_range[0] if score_range else "unknown"

def main():
    # Load training data
    train_matrix = Matrix("Train.dat")
    root = Tree().build_tree(train_matrix)
    
    print("Decision Tree Structure:")
    root.print_tree()

    # Load test data
    test_matrix = Matrix("Test.dat")
    test_scores = root.test_tree(test_matrix)
    original_scores = test_matrix.get_scores()

    print("\nOriginal_Scores (from Test.dat):")
    print("  ".join(original_scores))
    
    print("\nPredicted_Scores (from Test.dat):")
    print("  ".join(test_scores))
    
    # Calculate accuracy
    correct = sum(1 for t, o in zip(test_scores, original_scores) if t == o)
    accuracy = 100.0 * correct / len(test_scores)
    print(f"\nAccuracy on test data: {accuracy}%")

if __name__ == "__main__":
    main() 