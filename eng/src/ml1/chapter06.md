# Chapter 6. GBDT & Explainability

In the United States, when a bank denies a loan, the law (the Equal Credit
Opportunity Act) requires it to give the applicant a **specific reason** —
"credit score too low," "debt-to-income ratio too high," and so on. But if
the bank's prediction model is a complex ensemble combining hundreds of
trees, it's not easy to answer precisely "why was this person denied." The
model gives a correct answer (deny/approve), but even the model itself
can't explain **why** in a single line.

## 6.1 Making Trees Stronger: GBDT

Last chapter's random forest built many trees **independently** and combined
them by majority vote. **GBDT** (Gradient Boosted Decision Trees) takes a
different strategy: trees are added one at a time, **sequentially**, and
each new tree targets whatever the predictions so far have gotten wrong
(the **residual**).

Goal: sequentially add \\(T\\) trees \\(f_1, \ldots, f_T\\) to build the
prediction \\(F_T(x) = \sum_{t=1}^T f_t(x)\\). When training the \\(t\\)-th
tree, we treat the **residual** of the predictions accumulated so far,
\\(r^{(i)} = y^{(i)} - F_{t-1}(x^{(i)})\\), as the new target, and train a
tree to predict that residual:

```python
def gbdt_fit(X, y, n_trees, learning_rate):
    trees = []
    predictions = [0.0] * len(y)  # F_0(x) = 0
    for t in range(n_trees):
        residuals = [y[i] - predictions[i] for i in range(len(y))]
        tree = fit_single_tree(X, residuals)  # train one tree on the residuals
        trees.append(tree)
        for i in range(len(y)):
            predictions[i] += learning_rate * tree.predict(X[i])
    return trees
```

The `learning_rate` (shrinkage) deliberately shrinks each tree's
contribution, so no single tree overfits and the trees instead learn
gradually, splitting the work among many — a role similar to Chapter 2's
gradient descent learning rate. In fact, this process of "taking small
steps toward the residual" can be viewed as gradient descent in function
space (hence the name **gradient** boosting). XGBoost and LightGBM are
libraries that push this idea to an extreme level of practical
optimization, and they remain among the most frequent winners of tabular
data competitions (Kaggle and similar) to this day.

## 6.2 Why Predicting the Residual Improves the Whole Model

If \\(F_{t-1}\\) is already doing reasonably well, the remaining error (the
residual) is smaller than the original \\(y\\). If the new tree reduces that
smaller error further, the overall prediction \\(F_t = F_{t-1} + \eta f_t\\)
has an error that's one notch smaller still. In theory, the training error
keeps decreasing the more trees you add — but in practice, you have to stop
once validation performance starts getting worse (overfitting), a technique
called early stopping.

## 6.3 The Problem of Being Accurate but Unexplainable

GBDT is powerful, but with hundreds of trees intertwined, it's hard to know
which features contributed how much to any one prediction. **SHAP**
(SHapley Additive exPlanations) borrows the **Shapley value** from game
theory — the answer to "when several people collaborate on an outcome, how
should the credit be fairly split among them?" — and uses it to decompose a
single model prediction precisely into "how much each feature contributed."

## 6.4 SHAP: Decomposing One Prediction Feature by Feature

The original Shapley value answers: "when several players participate in a
cooperative game, what is each player's fair share?" — averaged over every
possible order in which players could join, based on the value each one
adds upon joining.

SHAP applies this by replacing "players" with "features": the SHAP value
\\(\phi_j\\) of feature \\(j\\) is the average, over every possible order in
which features are added one by one, of "how much the prediction changed
when feature \\(j\\) was added." The key property (**additivity**):

\\[f(x) = \phi_0 + \sum_{j=1}^n \phi_j\\]

Here \\(\phi_0\\) is the baseline (the average prediction over the whole
dataset), and \\(\phi_j\\) is how much feature \\(j\\) pushed the prediction
up (positive) or down (negative) from that baseline. SHAP's core guarantee
is that the sum of all \\(\phi_j\\) for one prediction is exactly equal to
the difference between the actual prediction and the baseline — giving an
exact-sum decomposition like "why was this loan denied: credit score
contributed -0.3, income contributed +0.1, ..."

**A small example**: with only 2 features (\\(A, B\\)), there are only two
possible orderings, \\(A \to B\\) and \\(B \to A\\):

\\[\phi_A = \frac{1}{2}\left[\big(f(\{A\})-f(\{\})\big) +
\big(f(\{A,B\})-f(\{B\})\big)\right]\\]

As the number of features grows, the number of possible orderings explodes
as \\(n!\\), so actual SHAP libraries approximate this average quickly via
sampling.

**A model's accuracy and its explainability are two separate axes — going
beyond "a model that predicts well" to "a model that can explain why it
predicted that way" is exactly the question SHAP is trying to answer.**

---

## Exercises

**1. (Coding)** Given a decision-stump function `fit_stump` (a depth-1
tree), complete `gbdt_fit` below (key lines left blank):

```python
def gbdt_fit(X, y, n_trees, learning_rate):
    trees = []
    predictions = [0.0] * len(y)
    for t in range(n_trees):
        # ADD ADDITIONAL CODE HERE!!
        # 1. compute residuals = y - predictions
        # 2. train one tree via fit_stump(X, residuals)
        # 3. accumulate predictions += learning_rate * tree's prediction

    return trees
```

**2. (Hand derivation, Tier C — fallback prepared)** A toy model with 2
features (\\(A, B\\)) has the prediction function \\(f(S)\\):

\\[f(\{\}) = 10, \quad f(\{A\}) = 16, \quad f(\{B\}) = 13, \quad f(\{A,B\}) = 20\\]

Find the marginal contribution of each feature for both possible orderings
(\\(A \to B\\), \\(B \to A\\)), and compute \\(\phi_A\\) and \\(\phi_B\\).
Verify that \\(\phi_0 + \phi_A + \phi_B = f(\{A,B\})\\) holds exactly.

**Fill-in-the-blank fallback version** (if free derivation is too
difficult):

```
Order A -> B: A's marginal contribution = f({A}) - f({}) = 16 - 10 = ______________
              B's marginal contribution = f({A,B}) - f({A}) = 20 - 16 = ______________
Order B -> A: B's marginal contribution = f({B}) - f({}) = 13 - 10 = ______________
              A's marginal contribution = f({A,B}) - f({B}) = 20 - 13 = ______________

phi_A = (A's contribution in A->B + A's contribution in B->A) / 2 = ______________
phi_B = (B's contribution in A->B + B's contribution in B->A) / 2 = ______________
Check: phi_0 + phi_A + phi_B = f({}) + phi_A + phi_B = ______________ (should equal 20)
```
