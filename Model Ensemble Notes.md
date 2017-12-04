# MODEL ENSEMBLING
Sorted by SP
[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)
[Original artical](https://mlwave.com/kaggle-ensembling-guide/)
## 1. Voting & Averaging
- Creating ensembles from submission files;
- Quick way to ensemble already existing model predictions;
-  Weighted majority vote -> give a better model more weight in a vote;
-  Averaging = Bagging submissions;
-  Averaging predictions often reduces overfit;
-  Rank averaging:
   1. Turn the predictions into ranks
   2. Averaging ranks
   3. Normalizing the averaged ranks between 0~1
- Historical ranks: For a new test sample, find the closest old predictions and take it's historical rank. 
## 2. Stacking/Blending
- Stacking = Blending ?;
>In some contexts, stacking is also referred to as blending^[Sill, Joseph, et al. "Feature-weighted linear stacking." arXiv preprint arXiv:0911.0460 (2009).]
- Basic idea: Using a pool of base classifiers, then using  another classifier to combine their predictions;
- Flow chart:
![Stacking flowchart](https://raw.githubusercontent.com/sam9211/samRep/master/Stacking_flowchart.png)[Original figure](http://7xlo8f.com1.z0.glb.clouddn.com/blog-diagram-stacking.jpg)
- Stacking with logistic regression (Python Code)
```python
# Variable:
#   X_train: training data
#   y_train: training labels
#   X_test : test data
#   clfs   : base classifiers
#   n_fold : number of folds
kf = KFold(n_fold, random_state=0)
# store predictions of left-out training data at each k-fold training procedure
stack_train = np.zeros((X_train.shape[0], len(clfs)))
# store predictions of test data of each base models
stack_test = np.zeros((X_test.shape[0], len(clfs)))
for i,clf in enumerate(clfs):
    # store predictions of test data at each k-fold training procedure
    kf_prd_i = np.zeros((X_test.shape[0], n_fold)) 
    for j,(train_idx, test_idx) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_idx]
        kf_y_train = y_train[train_idx]
        kf_X_test = X_train[test_idx]
        kf_y_test = y_trian[test_idx]
        clf.fit(kf_X_train, kf_y_train)
        kf_prd_i[:, j] = clf.predict_proba(X_test)[1]
        stack_train[test_idx, i] = clf.predict_proba(kf_X_test)[1]
    stack_test[:, i] = kf_prd_i.mean(axis=1)
#Level 2 classifier (Logistic regression)
l2_clf = logisticRegression().fit(stack_train, y_trian)
y_submission = clf.predict(stack_test)
```
- Stacking with non-linear algorithms
     + Popular algorithms: GBM, KNN, NN, RF, ET..
- Feature weighted linear stacking
     + Standard linear regression stacking:
     ![linear stacking](https://raw.githubusercontent.com/sam9211/samRep/master/linear%20stacking.png)
     + FWLS:
     ![FWLS](https://github.com/sam9211/samRep/blob/master/FWLS.png?raw=true)

     |feature space|features |discriptions|
     |-------------|---------|-------------|
     | f-feature space | f_1,f_2 | meta-data functions|
     | s-feature space | SVD, K-NN,RBM | learned prediction funtions|
     | quadratic features|SVD\*f_1, SVD\*f_2, K-NN \* f_1, K-NN \*f_2, RBM\*f_1, RBM\*f_2|fearture weighted|
- Quadratic linear stacking of models
     + Similar to FWLS, just creats combinations of model predictions.
- Stacking clsssifiers with regressors and vice versa
- Stacking unsupervised learned features
- Online stacking
- Model selection

## 3. Bagging
- Bagging-Boostrap aggregating, an ensemble meta-algorithm.
- Reduce variance and helps to avoid overfitting
- The idea:
     + Creat N boostrap samples {S_1,...,S_N} of S as follows:
         + For each S_i, randomly draw |S| examples from S with replacement
     + For each i = 1,...,N:
         + h_i = Learn(S_i)
     + Output H=<{h_1,...,h_n}, MajorityVote>
## 4. Boosting
- Key differences with respect to bagging:
     + Bagging: 
         Each individual classifiers is independent
     + Boosting: 
         1. Successive classifiers depend on their predecessors
         2. Place more weight on "hard" examples(i.e., instances that were misclassified on previous iterations)
- The idea:
     1. Equal weights are assigned to each training instance at first round (1/N for round 1)
     2. After C_i is learned, the weights are adjusted to allow the subsequent classifier C_i+1 to "pay more attention" to data that were miscalssified by C_i
     3. Final boosted classifier C* combines the votes of each individual classifier
- Adaboost (Adaptive Boost)
     + 
    
    
