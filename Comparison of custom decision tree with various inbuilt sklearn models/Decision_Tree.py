import numpy as np
import pandas as pd
from collections import deque

class Node:
    def __init__(self, elements:list[tuple], depth=0):
        self.feature = None
        self.midpoint = None
        self.depth = depth
        self.elements = elements
        self.gini_index = self.gini()
        self.right = None
        self.left = None
    def set_rule(self, feature, midpoint):
        self.feature = feature
        self.midpoint = midpoint
    def probabilites(self) -> dict[int, float]:
        classes_ = {}
        for p, c in self.elements:
            if c in classes_:
                classes_[c] += 1
            else:
                classes_[c] = 1
        probs = {}
        for c in classes_:
            probs[c] = classes_[c]/len(self.elements)
        return probs
    def gini(self) -> float:
        value = 1
        probs = self.probabilites()
        for p in probs.values():
            value -= p**2
        return value
    def predict(self):
        classes_ = {}
        for p, c in self.elements:
            if c in classes_:
                classes_[c] += 1
            else:
                classes_[c] = 1
        max_v = 0
        class_ = None
        for cl in classes_:
            if classes_[cl] > max_v:
                max_v = classes_[cl]
                class_ = cl
        return class_
    def split(self, factor):
        elements = list(map(lambda x: x[0][factor], self.elements))
        elements = sorted(elements)
        midpoints = []
        for i in range(1,len(elements)):
            p0 = elements[i-1]
            p1 = elements[i]
            midpoints.append((p0+p1)/2)
        max_gain = 0
        l, r = None, None
        best_midpoint = None
        for m in midpoints:
            right = []
            left = []
            for p, c in self.elements:
                if p[factor] < m:
                    right.append((p,c))
                else:
                    left.append((p,c))
            right_tree = Node(right, self.depth+1)
            left_tree = Node(left, self.depth+1)
            g1, g2  = right_tree.gini_index, left_tree.gini_index
            impurity = (len(right_tree.elements)*g1 + len(left_tree.elements)*g2)/len(self.elements)
            gain = self.gini_index - impurity
            if gain > max_gain:
                max_gain = gain
                r, l = right_tree, left_tree
                best_midpoint = m
        return best_midpoint, max_gain, l, r

class Tree:
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth
    def predict(self, X:pd.DataFrame|np.ndarray):
        predictions = []
        if isinstance(X, pd.DataFrame):
            features = list(X.columns)
            for i in range(len(X)):
                predictions.append(self.predict_single(X.iloc[i]))
        else:
            features = list(range(len(X[0])))
            for i in range(len(X)):
                predictions.append(self.predict_single(X[i]))
        return predictions
    def predict_single(self, x):
        curr = self.root
        while curr.feature is not None:
            if x[curr.feature] < curr.midpoint:
                curr = curr.right
            else:
                curr = curr.left
        return curr.predict().item()
    def fit(self, X: pd.DataFrame|np.ndarray, Y: pd.Series|np.ndarray):
        elements = []
        features = []
        if isinstance(X, pd.DataFrame):
            features = list(X.columns)
            for (i, r), y in zip(X.iterrows(), Y):
                elements.append((r, Y[i]))
        else:
            features = list(range(len(X[0])))
            for i, r in range(len(X)):
                elements.append((X[i], Y[i]))

        self.root = Node(elements)
        stack = deque()
        stack.append(self.root)
        while stack:
            n:Node = stack.pop()
            if n is None or n.depth == self.max_depth:
                continue
            best_gain = 0
            best_midpoint = None
            bl, br = None, None
            best_feature = None
            for feature in features:
                m, g, l,r = n.split(feature)
                if g <= 0:
                    continue
                if g > best_gain:
                    best_gain = g
                    best_midpoint = m
                    bl, br = l, r
                    best_feature = feature
            n.feature, n.midpoint = best_feature, best_midpoint
            n.left, n.right = bl, br
            stack.append(bl); stack.append(br)

class CustomDecisionTree:
    def __init__(self, max_depth=5):
        self.tree = Tree(max_depth=max_depth)
        self.classes_ = None
    def fit(self, X, y):
        self.tree.fit(X,y)
        self.classes_ = X.columns
    def predict(self, X):
        return self.tree.predict(X)