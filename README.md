# Decision Tree

The Decision Tree is one of the most commonly used machine learning models. It can be used as a classifier, similar to logistic regression, but unlike logistic regression, which performs a single-level classification, a decision tree works through multiple levels of decision-making.

At each level, the data is split based on specific feature-based criteria — these points of division are known as nodes. This hierarchical structure allows the model to make decisions step by step, leading to more refined and interpretable results.

<p align="center">
<img src="Images/decision_tree_model.webp" alt="decision_tree" width="50%"/>
</p>

Decision trees can be used to solve both regression and classification problems. The main difference lies in how the leaf nodes (the final nodes in the hierarchy) determine the output:

- For regression, each leaf node represents the average value of the target variable for the samples in that node.
- For classification, each leaf node represents the class with the highest number of votes (i.e., the majority class among the samples in that node).

**Since we already have regression and classification models, why do we need decision trees?**

Traditional models like regression and classification require several assumptions and preprocessing steps — such as checking for multicollinearity, ensuring feature scaling, and maintaining linear relationships between variables.

However, decision trees do not rely on these assumptions. They can naturally handle non-linear relationships, unclean or unscaled data, and correlated features without much preprocessing. This makes them easier to use, more flexible, and often more interpretable compared to traditional regression or classification models.

## Characteristics of Decision Trees

- **Highly interpretable** : The prediction process is easy to understand and visualize, as it follows a clear rule-based structure.
- **Handles all types of data** : Works well with both numerical and categorical data, with little to no need for preprocessing such as scaling or normalization.
- **Fast and efficient** : Training and prediction are computationally efficient, especially for small to medium-sized datasets.
- **Captures complex relationships** : Can model non-linear and complex feature interactions effectively.
- **Sensitive to data changes** : Decision trees are somewhat volatile — small changes in data can lead to a completely different tree structure.


## Important concepts for Decision Tree :

Initially, every decision tree starts with a root node that contains the entire dataset. From this point, the data is split into smaller subsets based on certain conditions. But how does this splitting actually happen? What criteria are used, and which features are chosen for each split? In this section, we’ll explore and address all these questions in detail.

## Homogeneity and Splitting

Let’s consider a dataset where we can split the data based on certain variables.
For example, if we split the dataset using the condition A > 10, we might end up with two groups—one containing 60% of one class and 40% of another. This seems like a reasonable split.
However, if we instead split on B > 10, we might get 5% of one class and 95% of another.

From this, two important questions arise:

- Which variable should we choose for splitting?
- Is it even necessary to split at this point?

To answer these, we introduce the concept of homogeneity, which measures how “pure” or similar the items in a group are. A highly homogeneous group means most of its elements belong to the same class.

In decision trees, data is split based on this homogeneity criterion—the goal is to make each resulting subset as pure (homogeneous) as possible.

<p align="center">
<img src="Images/splitting_homogeneity.webp" alt="homogeneity" width="50%"/>
</p>

There are several impurity measures used to calculate the homogeneity (or purity) of a node or split in a decision tree. These measures help determine how well a particular feature separates the data. The most commonly used impurity metrics are:

- Classification Error – Measures the fraction of incorrectly classified samples in a node.
- Gini Index – Measures the probability of incorrectly classifying a randomly chosen element.
- Entropy – Measures the level of uncertainty or randomness in the data; higher entropy means more disorder (less homogeneity).

Each of these metrics helps the algorithm decide the best feature and threshold for splitting to create the most homogeneous child nodes.

### Classification Error:

The Classification Error is one of the simplest measures used to evaluate the impurity of a node in a decision tree. It represents the proportion of samples that do not belong to the majority class within that node.

The formula is given by:

  $$
  \text{Classification Error (E) = 1 - max(p)}
  $$

where p is the probability of each class in the node.

Example:
Suppose we have a dataset of 100 samples. After a split, we get:
- 40 samples of one class
- 60 samples of another class

We can compute the class probabilities as: 

$$
\text{p(40)} = \frac{40}{100} = 0.4
$$
$$
\text{p(60)} = \frac{60}{100} = 0.6
$$
$$
\text{E}  = 1 - max(p) = 1 - 0.6 = 0.4
$$

### Gini Index

The Gini Index (or Gini Impurity) measures how often a randomly chosen sample would be incorrectly classified if it were labeled according to the class distribution in a node.
A lower Gini value indicates higher purity (more homogeneous node).

The formula is:

$$
Gini(G) = 1 -  \sum_{i=1}^n p_i^2
$$

Example : 
We take the same example as previous

$$
  Gini(G) = 1 -  (p(40)^2 + p(60)^2)
$$
$$
  Gini(G) = 1 -  ((0.4)^2 + (0.6)^2) = 1 - (0.16+0.36) = 0.48
$$

### Entropy
The Entropy metric measures the amount of disorder or randomness in a node.
A perfectly pure node (all samples belong to one class) has entropy = 0, while a completely mixed node has the highest entropy.

The formula is:

$$
Entropy(D) =  -  \sum_{i=1}^n p_i log_2 (p_i)
$$

Example : 
We take the same example as previous

$$
Entropy(D) =  - p(40)log_2 p(40) - p(60)log_2 p(60)
$$
$$
Entropy(D) =  - 0.4log_2 0.4 - 0.6log_2 0.6 = - 0.4 \times (-1.32) - 0.6 \times (-0.737) = 0.971
$$

<p align="center">
<img src="Images/impurity_measures.webp" alt="impurity_measure" width="50%"/>
</p>

Now, selecting one of these impurities, we can decide whether to split the node:

Homogeneity (H) =
- Classification Error (E) = 0.4 → If the calculated E for the current node is greater than the defined threshold (in our case 0.5), then split the node.
- Gini Index (G) = 0.48 → If the calculated G for the current node is greater than the defined threshold (e.g., 0.4), then split the node.
- Entropy (D) = 0.971 → If the calculated H for the current node is greater than the defined threshold (e.g., 0.5), then split the node.

So far, we have only discussed when to split a node. But what about which variable to split on? How do we decide which feature or attribute should be used to perform the split?

This question leads us to the next important step in building a decision tree: selecting the best splitting variable based on impurity reduction.

### Variable/ Attribute selection for split

The importance of a variable is measured using the Information Gain factor.

Information Gain represents the reduction in impurity achieved by splitting a node on a particular variable.
- If the impurity decreases significantly after the split, the information gain is high, indicating that the variable is effective at creating more homogeneous child nodes.
- In other words, the higher the information gain, the better the variable is at reducing impurity and making the split meaningful.

Formula :

$$ 
Δimpurity = (Pre-split Impurity) - (Post-split Impurity)
$$
$$ 
Gain = D - D_A
$$

post-split impurity is calculated by the weighted averagre of the two child nodes

Example : 

<p align="center">
<img src="Images/spliting_feature.webp" alt="splitting_feature" width="50%"/>
</p>

Lets see the left tree information gain, 

f1 entropy : 

$$
Entropy(D) =  - \frac{9}{14}log_2 \frac{9}{14} - \frac{5}{14}log_2 \frac{5}{14} = 0.94
$$

c1 entropy : 

$$
Entropy(D) =  - \frac{6}{8}log_2 \frac{6}{8} - \frac{2}{8}log_2 \frac{2}{8} = 0.81
$$

c2 entropy : 

$$
Entropy(D) =  - \frac{3}{6}log_2 \frac{3}{6} - \frac{3}{6}log_2 \frac{3}{6} = 1
$$

Information Gain

$$
Gain = D - D_A 
$$

$$
D_A = \frac{8}{14} \times c1_{entropy} + \frac{6}{14} \times c2_{entropy}
$$

$$
D_A = \frac{8}{14} \times 0.81 + \frac{6}{14} \times 1
$$

$$
D_A = 0.8924
$$

$$
Gain =  0.94 -  0.89 = 0.05
$$

right side tree information gain, 

f2 entropy : 

$$
Entropy(D) =  - \frac{8}{12}log_2 \frac{8}{12} - \frac{4}{12}log_2 \frac{4}{12} = 0.92
$$


c3 entropy : 

$$
Entropy(D) =  - \frac{3}{3}log_2 \frac{3}{3} = 0
$$

c4 entropy : 

$$
Entropy(D) =  - \frac{5}{9}log_2 \frac{5}{9} - \frac{4}{9}log_2 \frac{4}{9} = 0.99
$$


Information Gain

$$
Gain = D - D_A 
$$

$$
D_A = \frac{3}{12} \times c3_{entropy} + \frac{9}{12} \times c4_{entropy}
$$

$$
D_A = \frac{3}{12} \times 0 + \frac{9}{12} \times 0.99
$$

$$
D_A = 0.74
$$

$$
Gain =  0.92 -  0.74 = 0.16
$$


Based on the calculated information gain, it is more effective to split the tree at the f2 node, as it provides a higher gain compared to f1.

Although we used different cases for illustration, in the actual dataset both f1 and f2 nodes have the same number of input records.

## Disadvantages of Decision Trees

- Decision trees tend to overfit the training data, as they attempt to perfectly classify all examples, which can reduce generalization to unseen data.
- Small changes or variations in the input data can lead to significant changes in the tree structure, making the model unstable

**Overfitting can be tackle using two stratagies**
- **Truncation** : Stop the tree growth early, before it becomes too complex. This is also known as pre-pruning, where certain stopping criteria (like minimum information gain or maximum depth) are applied during the tree-building process.

- **Pruning** : Allow the tree to grow fully, then remove branches that contribute little to predictive power. This is done in a bottom-up manner, starting from the leaves, and is known as post-pruning.

Apart from these techniques, there are several parameters in the DecisionTreeClassifier model that can be adjusted to fine-tune its performance. This process is known as hyperparameter tuning.

### Key Hyperparameters in DecisionTreeClassifier

- **criterion (Gini / Entropy)** : Measures split quality. Default is "gini"; use "entropy" for information gain.
- **max_features** : Number of features to consider at each split. Can be an integer, float (percentage), "sqrt", "log2", or None (default = all features).
- **max_depth** : Maximum depth of the tree. Integer value or None (default = grow until leaves are pure or min_samples_split is reached).
- **min_samples_split** : Minimum samples needed to split a node. Integer or float (fraction). Default = 2.
- **min_samples_leaf** : Minimum samples required at a leaf. Integer or float (fraction). Default = 1.

