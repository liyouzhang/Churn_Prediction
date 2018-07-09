import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence, partial_dependence
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
import matplotlib.pyplot as plt

from feature_engineering import *

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
                'max_depth': [2,3,4],
                'max_features': ['auto',4]
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

def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, labels):
    n_obs = float(len(labels))
    thresholds = np.arange(0,1,0.01)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append([threshold_profit,threshold])
    return profits



def plot_model_profits(profits, save_path=None):
    threshold = []
    profit = []
    for p in profits:
        threshold.append(p[1])
        profit.append(p[0])
    plt.figure(figsize=(8,6))
    plt.plot(threshold, profit)
    plt.title("Profit Curve")
    plt.xlabel("TPR-FPR Threshold")
    plt.ylabel("Profit ($/user)")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    print("$ READ FILES")
    churnTrainDF = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
    y = build_y(churnTrainDF)
    df = feature_engineering(churnTrainDF)
    X = build_X(df)

    churnTestDF = pd.read_csv('data/churn_test.csv',parse_dates=['last_trip_date','signup_date'])
    y_test = build_y(churnTestDF)
    df2 = feature_engineering(churnTestDF)
    X_test = build_X(df2)


    print("$ Modeling")
    predictedLogistic = logisticModel(X, y, X_test)
    predictedRF = randomForestModel(X, y, X_test)
    predictedGradient = GradientBoost(X,y, X_test)
    predictedAda = adaBoost(X, y, X_test)

    # Plot the ROC curves for different models
    print("$ ROC curves")
    fpr_l, tpr_l, thresholds_l = roc_curve(y_test, predictedLogistic)
    auc_score_l = auc(fpr_l, tpr_l)
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, predictedRF)
    auc_score_rf = auc(fpr_rf, tpr_rf)
    fpr_a, tpr_a, thresholds_a = roc_curve(y_test, predictedAda)
    auc_score_a = auc(fpr_a, tpr_a)
    fpr_g, tpr_g, thresholds_g = roc_curve(y_test, predictedGradient)
    auc_score_g = auc(fpr_g, tpr_g)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_l, tpr_l, label= ('Logistic, AUC: {0:.2f}'.format(auc_score_l)))
    plt.plot(fpr_rf, tpr_rf, label=('Random Forest, AUC: {0:.2f}'.format(auc_score_rf)))
    plt.plot(fpr_a, tpr_a, label=('AdaBoosting, AUC: {0:.2f}'.format(auc_score_a)))
    plt.plot(fpr_g, tpr_g, label=('GradientBoosting, AUC: {0:.2f}'.format(auc_score_g)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc.png')
    plt.close()


    # Plot PDPs
    print("$ PDP Plots")
    clf = GradientBoostingRegressor(learning_rate=0.5, n_estimators=200, max_depth=3)
    fit = clf.fit(X, y)

    names = ['avg_dist_log', 'avg_rating_by_driver_log', 'avg_rating_of_driver_log',
       'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct',
       "city_King's Landing", 'city_Winterfell', 'phone_iPhone',
       'luxury_car_user_True', 'user_lifespan', 'user_rated_driver',
       'user_rated_driver_avg_rating_of_driver']
    features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

    pdp = plt.figure(figsize=(16,8))
    fig, axs = plot_partial_dependence(fit, X, features, feature_names=names, n_jobs=2, grid_resolution=50)
    fig.set_size_inches(18,12)
    fig.tight_layout()
    fig.savefig('pdp.png')
    plt.close(pdp)

    # plot profit curves
    cost_benefit = np.array([[20, -20], [0, 0]])
    profits = profit_curve(cost_benefit,predictedGradient,y_test)
    max_profit = max(profits)[0]
    max_threshold = max(profits)[1]
    print(max_profit, max_threshold)

    y_predict = predictedGradient >= max_threshold
    confusion_matrix = standard_confusion_matrix(y_test,y_predict)
    print('confusion matrix',confusion_matrix)
    threshold_profit = np.sum(confusion_matrix * cost_benefit) / y_test.shape[0]
    print('profit',threshold_profit)
    plot_model_profits(profits,save_path='profit.png')
