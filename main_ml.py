import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
import matplotlib.pyplot as plt

from feature_engineering_ml import *

def logisticModel(Xtrain, ytrain):


    # logReg = LogisticRegression()
    # fitted = logReg.fit(X, y)
    predictedLogistic = cross_val_predict(LogisticRegression(), Xtrain, ytrain, cv=3,method='predict_proba')[:,1]
    return predictedLogistic

def randomForestModel(Xtrain, ytrain):
    parameters = {'class_weight':['balanced', None],
                'max_depth': [2,3,4],
                'max_features': ['auto',4]
                }
    gscv = GridSearchCV(RandomForestClassifier(), parameters)
    fit = gscv.fit(Xtrain, ytrain)
    print('Best parameters: {}'.format(fit.best_params_))
    predictedRF = fit.predict_proba(Xtrain)[:,1]
    return predictedRF


def adaBoost(Xtrain, ytrain):
    parameters = {'learning_rate':[0.5,1.0],
                    'n_estimators':  [200,300]
                    }
    decisionTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(Xtrain, ytrain)
    print('Best parameters: {}'.format(fit.best_params_))
    predictedAda = fit.predict_proba(Xtrain)[:,1]
    return predictedAda

def adaBoost_predict(Xtrain, ytrain):
    parameters = {'learning_rate':[0.5,1.0],
                    'n_estimators':  [200,300]
                    }
    decisionTree = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
    gscv = GridSearchCV(decisionTree, parameters,scoring = 'roc_auc')
    fit = gscv.fit(Xtrain, ytrain)
    print('Best paremters: {}'.format(fit.best_params_))
    predictedAda = fit.predict(Xtrain)
    return predictedAda


def roc(yTrue, yHat):
    fpr, tpr, thresholds = roc_curve(yTrue, yHat)
    print("fpr ", fpr)
    print("tpr ", tpr)
    print("thresholds ", thresholds)
    auc_score = auc(fpr, tpr)
    print('auc score:', auc_score)
    plt.plot(fpr, tpr, 'b')
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# def profit_curve(yTrue,yHat):
#     [[tn, fp], [fn, tp]] = confusion_matrix(yTrue, yHat[:,1])
#     cost_benefit = np.array([[6, -3], [0, 0]])
#     thresholds = [1.0, 0.6, 0.4, 0.2]
#     confusion_matrices = [[tn, fp], [fn, tp]]
#     total_observations = len(yTrue)

#     for threshold, confusion_matrix in zip(thresholds, confusion_matrices):
#         threshold_expected_profit = np.sum(cost_benefit * confusion_matrix) / total_observations
#         print('Profit at threshold of {}: {}'.format(threshold, threshold_expected_profit))

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    # print(np.array([[tp, fp], [fn, tn]]))
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    # maybe_one = [] if 1 in predicted_probs else [1]
    # thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    thresholds = np.arange(0,1,0.01)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append([threshold_profit,threshold])
    # profits = ((np.array(profits), np.array(thresholds)))
    return profits



def plot_model_profits(profits, save_path=None):
    """Plotting function to compare profit curves of different models.
    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    
    # percentages = np.linspace(0, 100, len(profits))
    plt.plot(profits[1], profits[0])

    plt.title("Profit Curves")
    # plt.xlabel("Percentage of test instances (decreasing by score)")
    # plt.ylabel("Profit")
    # plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    print("$ READ FILES")
    churnTrainDF = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
    # X = churnTrainDF[['avg_dist', 'avg_surge']]
    # y = churnTrainDF['luxury_car_user']*1
    y = build_y(churnTrainDF)
    df = feature_engineering(churnTrainDF)
    # X = build_X(df,model="not logistic")
    X = build_X(df)

    # predictedLogistic = logisticModel(X, y)
    # roc(y, predictedLogistic)

    # predictedRF = randomForestModel(X, y)
    # roc(y, predictedRF)

    predictedAda = adaBoost(X, y)
    roc(y, predictedAda)
    predictedAda_label = adaBoost_predict(X,y)


    cost_benefit = np.array([[7, -3], [0, 0]])
    profits = profit_curve(cost_benefit,predictedAda,y)
    max_index = np.argmax(profits[0])
    print(max_index)

    best_threshold = profits[1][max_index]
    print('threshold', best_threshold)

    y_predict = predictedAda >= 0.5
    confusion_matrix = standard_confusion_matrix(y, predictedAda_label)
    print('confusion matrix',confusion_matrix)
    threshold_profit = np.sum(confusion_matrix * cost_benefit) / 40000
    print('profit',threshold_profit)
    plot_model_profits(profits)
