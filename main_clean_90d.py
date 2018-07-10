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
from models import *
from costbenefit_analysis import *

if __name__ == '__main__':
    print("$ Reading Data")
    churnTrainDF = pd.read_csv('data/churn_train.csv',parse_dates=['last_trip_date','signup_date'])
    churnTestDF = pd.read_csv('data/churn_test.csv',parse_dates=['last_trip_date','signup_date'])


    n_days = '90 days'
    y = build_y(churnTrainDF, delta_days=n_days)
    X = build_X(churnTrainDF, delta_days=n_days)
    y_test = build_y(churnTestDF, delta_days=n_days)
    X_test = build_X(churnTestDF, delta_days=n_days)

    print("$ Modeling")
    predictedLogistic = logisticModel(X, y, X_test)
    predictedRF = randomForestModel(X, y, X_test)
    predictedGradient = GradientBoost(X,y, X_test)
    predictedAda = adaBoost(X, y, X_test)
    predicted_G_L = GBC_Logistic(X,y,X_test)
    predicted_RF_L = RF_Logistic(X,y,X_test)

    # Plot the ROC curves for different models
    print("$ ROC Curves")
    fpr_l, tpr_l, thresholds_l = roc_curve(y_test, predictedLogistic)
    auc_score_l = auc(fpr_l, tpr_l)
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, predictedRF)
    auc_score_rf = auc(fpr_rf, tpr_rf)
    fpr_a, tpr_a, thresholds_a = roc_curve(y_test, predictedAda)
    auc_score_a = auc(fpr_a, tpr_a)
    fpr_g, tpr_g, thresholds_g = roc_curve(y_test, predictedGradient)
    auc_score_g = auc(fpr_g, tpr_g)
    fpr_g_l, tpr_g_l, thresholds_g_l = roc_curve(y_test, predicted_G_L)
    auc_score_g_l = auc(fpr_g_l, tpr_g_l)
    fpr_rf_l, tpr_rf_l, thresholds_rf_l = roc_curve(y_test, predicted_RF_L)
    auc_score_rf_l = auc(fpr_rf_l, tpr_rf_l)
    

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_l, tpr_l, label= ('Logistic, AUC: {0:.2f}'.format(auc_score_l)))
    plt.plot(fpr_rf, tpr_rf, label=('Random Forest, AUC: {0:.2f}'.format(auc_score_rf)))
    plt.plot(fpr_a, tpr_a, label=('AdaBoosting, AUC: {0:.2f}'.format(auc_score_a)))
    plt.plot(fpr_g, tpr_g, label=('GradientBoosting, AUC: {0:.2f}'.format(auc_score_g)))
    plt.plot(fpr_g_l, tpr_g_l, label=('GBM + LG, AUC: {0:.2f}'.format(auc_score_g_l)))
    plt.plot(fpr_rf_l, tpr_rf_l, label=('RF + LG, AUC: {0:.2f}'.format(auc_score_rf_l)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('{} ROC curve'.format(n_days))
    plt.legend(loc='best')
    plt.savefig('{} roc.png'.format(n_days))
    plt.close()


    # Plot PDPs
    print("$ PDP Plots")
    clf = GradientBoostingClassifier(learning_rate=0.5, n_estimators=200, max_depth=3)
    fit = clf.fit(X, y)

    names = ['avg_dist_log', 'avg_rating_by_driver_log', 'avg_rating_of_driver_log',
       'avg_surge', 'surge_pct', 'trips_in_first_30_days', 'weekday_pct',
       "city_King's Landing", 'city_Winterfell', 'phone_iPhone',
       'luxury_car_user_True', 'user_lifespan', 'user_rated_driver',
       'user_rated_driver_avg_rating_of_driver', 'city_Astapor', 'phone_Android',
        'luxury_car_user_False', 'avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver']

    features = [5,9,15,3,4,6,7,8,14,16,10,11,17,18,19]

    pdp = plt.figure(figsize=(16,8))
    # plt.title('{} Partial Dependency Plot'.format(n_days))
    fig, axs = plot_partial_dependence(fit, X, features, feature_names=names, n_jobs=2, grid_resolution=50)
    fig.set_size_inches(18,12)
    fig.tight_layout()
    fig.savefig('{} pdp.png'.format(n_days))
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

    plot_model_profits(profits,save_path='{} profit.png'.format(n_days),n_days=n_days)
