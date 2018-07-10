from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

def logisticModel(Xtrain, ytrain, X_test):
    # logReg = LogisticRegression()
    # fitted = logReg.fit(X, y)
    # predictedLogistic = cross_val_predict(LogisticRegression(), Xtrain, ytrain, cv=3,method='predict_proba')[:,1]
    lg = LogisticRegression()
    fit = lg.fit(Xtrain,ytrain)
    predictedLogistic = fit.predict_proba(X_test)[:,1]
    return predictedLogistic

def randomForestModel(Xtrain, ytrain,X_test):
    parameters = {'class_weight':['balanced', None],
                'max_depth': [2,4],
                'max_features': ['auto',4,5]
                }
    gscv = GridSearchCV(RandomForestClassifier(), parameters)
    fit = gscv.fit(Xtrain, ytrain)
    print('Best parameters for RF: {}'.format(fit.best_params_))
    predictedRF = fit.predict_proba(X_test)[:,1]
    return predictedRF

def GradientBoost(Xtrain, ytrain,X_test):
    parameters = {'learning_rate':[0.5,1.0],
                    'n_estimators':  [200,300]
                    }
    decisionTree = GradientBoostingClassifier()
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(Xtrain, ytrain)
    print('Best parameters for GBM: {}'.format(fit.best_params_))
    predictedGradient = fit.predict_proba(X_test)[:,1]
    return predictedGradient

def adaBoost(Xtrain, ytrain,X_test):
    parameters = {'learning_rate':[0.5,1.0],
                    'n_estimators':  [200,300]
                    }
    decisionTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(Xtrain, ytrain)
    print('Best parameters for ABM: {}'.format(fit.best_params_))
    predictedAda = fit.predict_proba(X_test)[:,1]
    return predictedAda