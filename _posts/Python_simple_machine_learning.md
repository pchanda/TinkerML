

```python
#Lets start with installing pandas from within python interactive shell. This may be needed if you are missing a package.
import pip
#if you have pip installed, you can use this to install any package you need : pip.main(['install',package-name])
pip.main(['install','pandas']) 
```

    Requirement already satisfied: pandas in c:\users\u588401\appdata\local\continuum\anaconda3\lib\site-packages
    Requirement already satisfied: python-dateutil>=2 in c:\users\u588401\appdata\local\continuum\anaconda3\lib\site-packages (from pandas)
    Requirement already satisfied: pytz>=2011k in c:\users\u588401\appdata\local\continuum\anaconda3\lib\site-packages (from pandas)
    Requirement already satisfied: numpy>=1.9.0 in c:\users\u588401\appdata\local\continuum\anaconda3\lib\site-packages (from pandas)
    Requirement already satisfied: six>=1.5 in c:\users\u588401\appdata\local\continuum\anaconda3\lib\site-packages (from python-dateutil>=2->pandas)
    

    You are using pip version 9.0.1, however version 10.0.1 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    




    0



Now lets use pandas to load the dataset from a url.


```python
import pandas
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset = pandas.read_csv(url)
print('shape=',dataset.shape)
print(dataset.head())
```

    shape= (149, 5)
       5.1  3.5  1.4  0.2  Iris-setosa
    0  4.9  3.0  1.4  0.2  Iris-setosa
    1  4.7  3.2  1.3  0.2  Iris-setosa
    2  4.6  3.1  1.5  0.2  Iris-setosa
    3  5.0  3.6  1.4  0.2  Iris-setosa
    4  5.4  3.9  1.7  0.4  Iris-setosa
    

Oh no, the dataset has no column names, lets add them


```python
dataset.columns=['A','B','C','D','E']
print(dataset.head())
```

         A    B    C    D            E
    0  4.9  3.0  1.4  0.2  Iris-setosa
    1  4.7  3.2  1.3  0.2  Iris-setosa
    2  4.6  3.1  1.5  0.2  Iris-setosa
    3  5.0  3.6  1.4  0.2  Iris-setosa
    4  5.4  3.9  1.7  0.4  Iris-setosa
    

Lets look at the class distribution of the column we will use as target for our machine learning tasks, column 'E'.


```python
print(dataset.groupby('E').size())
```

    E
    Iris-setosa        49
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64
    

Now lets split our dataset randomly into training, testing, and validation datasets. First, shuffle the dataset randomly with replacement.


```python
# The frac keyword argument specifies the fraction of rows to return in
# the random sample, so frac=1 means return all rows (in random order).
dataset_shuffled = dataset.sample(frac=1)
print(dataset_shuffled.head())
```

           A    B    C    D                E
    9    5.4  3.7  1.5  0.2      Iris-setosa
    113  5.8  2.8  5.1  2.4   Iris-virginica
    96   6.2  2.9  4.3  1.3  Iris-versicolor
    118  6.0  2.2  5.0  1.5   Iris-virginica
    147  6.2  3.4  5.4  2.3   Iris-virginica
    

Now split the data into training and validation sets in 80:20 ratio. We will use the validation set to test the 
performance of machine learning algorithms.


```python
n = dataset_shuffled.shape[0]
validation_size = int(0.2*n)

from sklearn import model_selection

data_array = dataset_shuffled.values
X = data_array[:,0:4]  # separate the predictor variables
Y = data_array[:,4]    # from the target variable (last column)
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=13)
```

Check the sizes of the training and validation data sets.


```python
print('Train_X = ',X_train.shape, 'Train_Y = ',Y_train.shape)
print('Val_X = ',X_validation.shape, 'Val_Y = ',Y_validation.shape)
```

    Train_X =  (120, 4) Train_Y =  (120,)
    Val_X =  (29, 4) Val_Y =  (29,)
    


```python
import sklearn
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import neighbors

# we will use these classifiers
LR = linear_model.LogisticRegression()
KNN = neighbors.KNeighborsClassifier()
DT = tree.DecisionTreeClassifier()
SVM = svm.SVC()

models_list = [LR,KNN,DT,SVM]
num_folds = 8
kfolds = model_selection.KFold(n_splits=num_folds, shuffle=True, random_state=13)
for model in models_list:
    model_name = type(model).__name__
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,scoring='accuracy',cv=kfolds)
    print(model_name,' : ',cv_results.tolist())
    print(model_name,' : mean=',cv_results.mean(), ' sd=',cv_results.std())
    print('------------------------------------------------------------------------')
```

    LogisticRegression  :  [1.0, 1.0, 1.0, 0.6666666666666666, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 1.0]
    LogisticRegression  : mean= 0.9333333333333333  sd= 0.10540925533894599
    ------------------------------------------------------------------------
    KNeighborsClassifier  :  [1.0, 0.9333333333333333, 1.0, 0.8666666666666667, 1.0, 0.9333333333333333, 1.0, 1.0]
    KNeighborsClassifier  : mean= 0.9666666666666667  sd= 0.04714045207910316
    ------------------------------------------------------------------------
    DecisionTreeClassifier  :  [0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 0.9333333333333333, 1.0, 0.9333333333333333, 1.0, 0.9333333333333333]
    DecisionTreeClassifier  : mean= 0.95  sd= 0.02886751345948128
    ------------------------------------------------------------------------
    SVC  :  [0.9333333333333333, 1.0, 0.9333333333333333, 0.9333333333333333, 1.0, 0.9333333333333333, 1.0, 1.0]
    SVC  : mean= 0.9666666666666667  sd= 0.033333333333333326
    ------------------------------------------------------------------------
    

But this is naive training with each of the classifiers as we have not done anything to optimize the parameters for each classifer. We will use sklearn.model_selection.GridSearchCV for that. Lets do this with each classifier separately because each classifier will have its own set of parametes to tune. 


```python
kfolds = model_selection.KFold(n_splits=num_folds,shuffle=True,random_state=13)

#define function to be called to do grid search for each of the classifiers.

def print_cv_results(estimator,results):
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    print(type(estimator).__name__)
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print('best_parameters=',results.best_params_)
    print('best_accuracy=',results.best_score_)
    print('----------------------------------------------')
    
def do_grid_search(estimator,grid_values,kfolds):
    clf = model_selection.GridSearchCV(estimator=estimator,param_grid=grid_values,scoring='accuracy',cv=kfolds)
    results = clf.fit(X_train,Y_train)
    print_cv_results(estimator,results)
    return clf
```


```python
# First, logistic Regression parameters to tune : C, penalty
grid_values = {'C':[0.01,0.1,1,10,100],'penalty':['l1','l2']}
clf_LR = do_grid_search(LR,grid_values,kfolds) 

# Next we do with KNN
grid_values = {'n_neighbors':list(range(1,10))}
clf_KNN = do_grid_search(KNN,grid_values,kfolds)

# Next Decision Tree
grid_values = {'criterion':['gini','entropy'],'splitter':['best','random'],'min_samples_split':list(range(2,10)),'min_samples_leaf':[1,2,3,4,5]}
clf_DT = do_grid_search(DT,grid_values,kfolds)
```

    LogisticRegression
    0.233 (+/-0.149) for {'C': 0.01, 'penalty': 'l1'}
    0.675 (+/-0.356) for {'C': 0.01, 'penalty': 'l2'}
    0.725 (+/-0.380) for {'C': 0.1, 'penalty': 'l1'}
    0.817 (+/-0.420) for {'C': 0.1, 'penalty': 'l2'}
    0.950 (+/-0.088) for {'C': 1, 'penalty': 'l1'}
    0.933 (+/-0.211) for {'C': 1, 'penalty': 'l2'}
    0.967 (+/-0.094) for {'C': 10, 'penalty': 'l1'}
    0.967 (+/-0.094) for {'C': 10, 'penalty': 'l2'}
    0.958 (+/-0.093) for {'C': 100, 'penalty': 'l1'}
    0.967 (+/-0.094) for {'C': 100, 'penalty': 'l2'}
    best_parameters= {'C': 10, 'penalty': 'l1'}
    best_accuracy= 0.9666666666666667
    ----------------------------------------------
    KNeighborsClassifier
    0.958 (+/-0.065) for {'n_neighbors': 1}
    0.950 (+/-0.058) for {'n_neighbors': 2}
    0.958 (+/-0.065) for {'n_neighbors': 3}
    0.967 (+/-0.067) for {'n_neighbors': 4}
    0.967 (+/-0.094) for {'n_neighbors': 5}
    0.975 (+/-0.065) for {'n_neighbors': 6}
    0.975 (+/-0.065) for {'n_neighbors': 7}
    0.975 (+/-0.065) for {'n_neighbors': 8}
    0.975 (+/-0.065) for {'n_neighbors': 9}
    best_parameters= {'n_neighbors': 6}
    best_accuracy= 0.975
    ----------------------------------------------
    DecisionTreeClassifier
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 3, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 3, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'best'}
    0.975 (+/-0.065) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
    0.900 (+/-0.115) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 6, 'splitter': 'best'}
    0.925 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 6, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 7, 'splitter': 'best'}
    0.967 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 7, 'splitter': 'random'}
    0.933 (+/-0.115) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 8, 'splitter': 'best'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 8, 'splitter': 'random'}
    0.933 (+/-0.115) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 9, 'splitter': 'best'}
    0.925 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 9, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}
    0.958 (+/-0.093) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 3, 'splitter': 'best'}
    0.925 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 3, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 4, 'splitter': 'best'}
    0.917 (+/-0.111) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 4, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'best'}
    0.975 (+/-0.065) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 6, 'splitter': 'best'}
    0.925 (+/-0.155) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 6, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 7, 'splitter': 'best'}
    0.967 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 7, 'splitter': 'random'}
    0.933 (+/-0.115) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 8, 'splitter': 'best'}
    0.917 (+/-0.111) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 8, 'splitter': 'random'}
    0.933 (+/-0.115) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 9, 'splitter': 'best'}
    0.958 (+/-0.093) for {'criterion': 'gini', 'min_samples_leaf': 2, 'min_samples_split': 9, 'splitter': 'random'}
    0.967 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 2, 'splitter': 'best'}
    0.933 (+/-0.149) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 2, 'splitter': 'random'}
    0.967 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'best'}
    0.958 (+/-0.093) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'random'}
    0.958 (+/-0.065) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 4, 'splitter': 'best'}
    0.917 (+/-0.173) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 4, 'splitter': 'random'}
    0.967 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'best'}
    0.958 (+/-0.093) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 6, 'splitter': 'best'}
    0.925 (+/-0.104) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 6, 'splitter': 'random'}
    0.958 (+/-0.065) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 7, 'splitter': 'best'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 8, 'splitter': 'best'}
    0.908 (+/-0.132) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 9, 'splitter': 'best'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 3, 'min_samples_split': 9, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'best'}
    0.908 (+/-0.114) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 4, 'splitter': 'best'}
    0.933 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 4, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 5, 'splitter': 'best'}
    0.950 (+/-0.058) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 5, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 6, 'splitter': 'best'}
    0.933 (+/-0.067) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 6, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 7, 'splitter': 'best'}
    0.875 (+/-0.294) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 8, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 9, 'splitter': 'best'}
    0.933 (+/-0.163) for {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 9, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'best'}
    0.875 (+/-0.194) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 3, 'splitter': 'best'}
    0.900 (+/-0.231) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 3, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 4, 'splitter': 'best'}
    0.925 (+/-0.104) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 4, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 5, 'splitter': 'best'}
    0.900 (+/-0.149) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 5, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 6, 'splitter': 'best'}
    0.925 (+/-0.155) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 6, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 7, 'splitter': 'best'}
    0.900 (+/-0.094) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 7, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 8, 'splitter': 'best'}
    0.900 (+/-0.163) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 8, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 9, 'splitter': 'best'}
    0.917 (+/-0.160) for {'criterion': 'gini', 'min_samples_leaf': 5, 'min_samples_split': 9, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'best'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'random'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 3, 'splitter': 'best'}
    0.892 (+/-0.114) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 3, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'best'}
    0.933 (+/-0.133) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'random'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
    0.933 (+/-0.094) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 6, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 6, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 7, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 8, 'splitter': 'best'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 9, 'splitter': 'best'}
    0.917 (+/-0.160) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 9, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 3, 'splitter': 'best'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 3, 'splitter': 'random'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 4, 'splitter': 'best'}
    0.925 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 4, 'splitter': 'random'}
    0.933 (+/-0.094) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'best'}
    0.958 (+/-0.093) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'random'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 6, 'splitter': 'best'}
    0.942 (+/-0.169) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 6, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 7, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 8, 'splitter': 'best'}
    0.950 (+/-0.058) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 9, 'splitter': 'best'}
    0.950 (+/-0.111) for {'criterion': 'entropy', 'min_samples_leaf': 2, 'min_samples_split': 9, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 2, 'splitter': 'best'}
    0.950 (+/-0.111) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 2, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'best'}
    0.900 (+/-0.189) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'random'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 4, 'splitter': 'best'}
    0.933 (+/-0.094) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 4, 'splitter': 'random'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'best'}
    0.925 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 5, 'splitter': 'random'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 6, 'splitter': 'best'}
    0.908 (+/-0.210) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 6, 'splitter': 'random'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 7, 'splitter': 'best'}
    0.908 (+/-0.132) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 8, 'splitter': 'best'}
    0.925 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 9, 'splitter': 'best'}
    0.908 (+/-0.162) for {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 9, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'best'}
    0.950 (+/-0.058) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 3, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 4, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 4, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 5, 'splitter': 'best'}
    0.942 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 5, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 6, 'splitter': 'best'}
    0.933 (+/-0.115) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 6, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 7, 'splitter': 'best'}
    0.933 (+/-0.133) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 7, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 8, 'splitter': 'best'}
    0.942 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 8, 'splitter': 'random'}
    0.942 (+/-0.124) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 9, 'splitter': 'best'}
    0.908 (+/-0.148) for {'criterion': 'entropy', 'min_samples_leaf': 4, 'min_samples_split': 9, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'best'}
    0.958 (+/-0.065) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 2, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 3, 'splitter': 'best'}
    0.917 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 3, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 4, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 4, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 5, 'splitter': 'best'}
    0.950 (+/-0.088) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 5, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 6, 'splitter': 'best'}
    0.900 (+/-0.115) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 6, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 7, 'splitter': 'best'}
    0.908 (+/-0.114) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 7, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 8, 'splitter': 'best'}
    0.925 (+/-0.080) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 8, 'splitter': 'random'}
    0.950 (+/-0.129) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 9, 'splitter': 'best'}
    0.925 (+/-0.104) for {'criterion': 'entropy', 'min_samples_leaf': 5, 'min_samples_split': 9, 'splitter': 'random'}
    best_parameters= {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 4, 'splitter': 'random'}
    best_accuracy= 0.975
    ----------------------------------------------
    


```python
# And finally SVM
grid_values = {'C':[1,10,100],'kernel':['linear', 'poly', 'rbf', 'sigmoid'] }
clf_SVM = do_grid_search(SVM,grid_values,kfolds)
```

    SVC
    0.975 (+/-0.065) for {'C': 1, 'kernel': 'linear'}
    0.958 (+/-0.093) for {'C': 1, 'kernel': 'poly'}
    0.967 (+/-0.067) for {'C': 1, 'kernel': 'rbf'}
    0.192 (+/-0.140) for {'C': 1, 'kernel': 'sigmoid'}
    0.967 (+/-0.094) for {'C': 10, 'kernel': 'linear'}
    0.950 (+/-0.088) for {'C': 10, 'kernel': 'poly'}
    0.967 (+/-0.094) for {'C': 10, 'kernel': 'rbf'}
    0.192 (+/-0.140) for {'C': 10, 'kernel': 'sigmoid'}
    0.958 (+/-0.093) for {'C': 100, 'kernel': 'linear'}
    0.942 (+/-0.080) for {'C': 100, 'kernel': 'poly'}
    0.942 (+/-0.104) for {'C': 100, 'kernel': 'rbf'}
    0.192 (+/-0.140) for {'C': 100, 'kernel': 'sigmoid'}
    best_parameters= {'C': 1, 'kernel': 'linear'}
    best_accuracy= 0.975
    ----------------------------------------------
    


```python
from sklearn.metrics import classification_report

# predictions for LR
Y_true, Y_pred = Y_validation, clf_LR.predict(X_validation)
print(classification_report(Y_true, Y_pred))
```

                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       1.00      1.00      1.00         9
     Iris-virginica       1.00      1.00      1.00        10
    
        avg / total       1.00      1.00      1.00        29
    
    


```python
# predictions for KNN
Y_true, Y_pred = Y_validation, clf_KNN.predict(X_validation)
print(classification_report(Y_true, Y_pred))
```

                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       0.90      1.00      0.95         9
     Iris-virginica       1.00      0.90      0.95        10
    
        avg / total       0.97      0.97      0.97        29
    
    


```python
# predictions for DT
Y_true, Y_pred = Y_validation, clf_DT.predict(X_validation)
print(classification_report(Y_true, Y_pred))
```

                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       0.82      1.00      0.90         9
     Iris-virginica       1.00      0.80      0.89        10
    
        avg / total       0.94      0.93      0.93        29
    
    


```python
# predictions for SVM
Y_true, Y_pred = Y_validation, clf_SVM.predict(X_validation)
print(classification_report(Y_true, Y_pred))
```

                     precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00        10
    Iris-versicolor       0.90      1.00      0.95         9
     Iris-virginica       1.00      0.90      0.95        10
    
        avg / total       0.97      0.97      0.97        29
    
    
