from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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


def GBC_Logistic(X_train,y_train,X_test):
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)
    grd = GradientBoostingClassifier(n_estimators=200,learning_rate=0.5)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    return y_pred_grd_lm
    # fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)

def RF_Logistic(X_train,y_train,X_test):
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)
    grd = RandomForestClassifier(max_depth=4,max_features=5)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    # print(grd.apply(X_train).shape)
    grd_enc.fit(grd.apply(X_train))
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)), y_train_lr)
    y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)))[:, 1]
    return y_pred_grd_lm
