# import relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# machine learning algorithms 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

# evaluation of algorithms 
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV

# resampling algorithms 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline


#######################################################
################### 1. Load Data ######################
#######################################################
filepath = r"diabetes_data.csv"
diab_df = pd.read_csv(filepath)

# read the first five rows of data
print(diab_df.head())


#######################################################################
################### 2. Investigate Missing Data  ######################
#######################################################################

# display the data information 
print(diab_df.info())

# check for missing data on respective column 
print(diab_df.isnull().sum())

# using heatmap to visualize for missing data on respective column 
sns.heatmap(diab_df.isnull(), annot = False, cmap = 'viridis')


########################################################################
################### 3. Exploratory Data Analysis  ######################
########################################################################

# age distribution with absence and presence of diabetes
sns.set_style("darkgrid")
sns.displot(x = "age", data = diab_df, hue = "class")
plt.title("Age Distribution with Diabetes and Non-Diabetes", fontweight = "bold", fontsize = 15)
plt.xlabel("Age", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.show()

# gender distribution with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "gender", data = diab_df, hue = "class")
plt.title("Gender Distribution with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Gender", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Class", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# polyuria Symptom with abensece and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "polyuria", data = diab_df, hue = "class")
plt.title("Polyuria Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Polyuria", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# polydipsia Symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "polydipsia", data = diab_df, hue = "class")
plt.title("Polydipsia Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Polydipsia", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# sudden weight loss symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "sudden_weight_loss", data = diab_df, hue = "class")
plt.title("Sudden Weight Loss Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Sudden Weight Loss", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# polyphagia symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "polyphagia", data = diab_df, hue = "class")
plt.title("Polyphagia Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Polyphagia", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# weakness symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "weakness", data = diab_df, hue = "class")
plt.title("Weakness Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Weakness", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# genital_thrush distribution with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "genital_thrush", data = diab_df, hue = "class")
plt.title("Genital Thrush Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Genital Thrush", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# visual blurring symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "visual_blurring", data = diab_df, hue = "class")
plt.title("Visual Blurring Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Visual Blurring", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# Itching symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "itching", data = diab_df, hue = "class")
plt.title("Itching Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Itching", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# Irritability symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "irritability", data = diab_df, hue = "class")
plt.title("Irritability Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Irritability", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# delayed healing symptom with adbsence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "delayed_healing", data = diab_df, hue = "class")
plt.title("Delayed Healing Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Delayed Healing", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# partial paresis symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "partial_paresis", data = diab_df, hue = "class")
plt.title("Partial Paresis Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Partial Paresis", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# muscle stiffness symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "muscle_stiffness", data = diab_df, hue = "class")
plt.title("Muscle Stiffness Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Muscle Stiffness", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# alopecia paresis symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "alopecia", data = diab_df, hue = "class")
plt.title("Alopecia Paresis Symptom with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Alopecia", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()

# obesity stiffness symptom with absence and presence of diabetes
sns.set_style("darkgrid")
sns.countplot(x = "obesity", data = diab_df, hue = "class")
plt.title("Obesity with Diabetes and Non-Diabetes", fontweight = 'bold', fontsize = 15)
plt.xlabel("Obesity", fontweight = 'bold', fontsize = 12)
plt.ylabel("Number of People", fontweight = 'bold', fontsize = 12)
plt.legend(bbox_to_anchor = (1.35, 0.6), title = "Diagnosis", shadow = True, edgecolor = "navy", labels = ['0: Non-Diabetes', '1: Diabetes'])
plt.show()


##################################################################
################### 4. Data Pre-Processing  ######################
##################################################################
# extract the column features
column_features = diab_df.drop(columns = ['class'], axis = 1).columns


# convert the gender column from categorical feature into numerical feature 
# define an object for label encoder
le = LabelEncoder()
# fit and transform the gender column from cat into num 
diab_df['gender'] = le.fit_transform(diab_df['gender'])

# standardized the data using min-max scaler
# define an object for min-max scaler
scaler = MinMaxScaler()
diab_df[column_features] = scaler.fit_transform(diab_df[column_features])


############################################################
################### 5.Data Splitting  ######################
############################################################
X_train, X_test, y_train, y_test = train_test_split(diab_df[column_features], 
                                                    diab_df['class'],
                                                    test_size = 0.2,
                                                    random_state = 42)

##############################################################
################### 6.Spot Check on ML  ######################
##############################################################
def get_ML_models():
    name_list, model_list = list(), list()
    
    # decision tree classifier
    dtc = DecisionTreeClassifier(random_state = 42)
    name_list.append('decision tree')
    model_list.append(dtc)
    
    # random forest classifier
    rfc = RandomForestClassifier(random_state = 42)
    name_list.append('random forest')
    model_list.append(rfc)
    
    # gradient boosting classifier
    gbc = GradientBoostingClassifier(random_state = 42)
    name_list.append('gradient boosting')
    model_list.append(gbc)
    
    # extra tree classifier
    etc = ExtraTreesClassifier(random_state = 42)
    name_list.append('extra trees')
    model_list.append(etc)
    
    # adaboost classifier
    ada = AdaBoostClassifier(random_state = 42)
    name_list.append('adaboost')
    model_list.append(ada)
    
    return name_list, model_list


def evaluate_model(model, X, y):
    # define cross validation procedure 
    cv = ShuffleSplit(train_size = 0.8, test_size = 0.2, n_splits = 10, random_state = 42)
    # define roc-auc-score
    roc_auc_scoring = make_scorer(roc_auc_score, needs_proba = True)
    # compute the cv score 
    score = cross_validate(model, X, y, cv = cv, n_jobs = 1, return_train_score = True, scoring = roc_auc_scoring)
    return score


# call the function to get the model list
names, models = get_ML_models()

for i in range(len(models)):
    # build and fit the model
    model = models[i]
    model.fit(X_train, y_train)
    
    # evaluate the model
    scoring = evaluate_model(model, X_train, y_train)
    print(">> {}: {:.3f} ({:.3f})".format(names[i], scoring['train_score'].mean(), scoring['train_score'].std()))


#################################################################################
################### 7.Spot Check on Resampling Techniques  ######################
#################################################################################
def get_resampling_models():
    name_list, model_list = list(), list()
    
    # smote
    smote = SMOTE(random_state = 42)
    name_list.append('smote')
    model_list.append(smote)
    
    # borderline smote
    borderline_smote = BorderlineSMOTE(random_state = 42)
    name_list.append('borderline smote')
    model_list.append(borderline_smote)
    
    # smote-tomek
    smote_tomek = SMOTETomek(random_state = 42)
    name_list.append('smote-tomek')
    model_list.append(smote_tomek)
    
    # smote-enn
    smote_enn = SMOTEENN(random_state = 42)
    name_list.append('smote-enn')
    model_list.append(smote_enn)
    
    return name_list, model_list

# define adaboost classifier
adaboostClassifier = AdaBoostClassifier(random_state = 42)

# call the function to get the resampling name, and model 
names_, models_ = get_resampling_models()

for i in range(len(models_)):
    # define pipeline steps
    steps = [('o', models_[i]), ('m', adaboostClassifier)]
    # create pipeline 
    pipe = Pipeline(steps = steps)
    # call the function to evaluate the score 
    scoring = evaluate_model(pipe, X_train, y_train)
    print(">> {}: {:.3f} ({:.3f})".format(names_[i], scoring['train_score'].mean(), scoring['train_score'].std()))


#######################################################################################
################### 8. SMOTE-Tomek on Imbalanced Classification  ######################
#######################################################################################
print("Before applying SMOTE-Tomek on Imbalanced Classification: ")
print(y_train.value_counts())

# distribution of imbalanced classification before applying resampling technique
y_train.value_counts().plot(kind = 'barh', color = ['skyblue', 'limegreen'])
plt.title("Distribution of Imbalanced Classification (Before Applying SMOTE-Tomek)", fontweight = 'bold', fontsize = 15)
plt.xlabel("Number of Observation", fontweight = 'bold', fontsize = 12)
plt.ylabel("Class", fontweight = 'bold', fontsize = 12)
plt.show()

# balance the classification using SMOTE-Tomek
smt = SMOTETomek()
x_train_res, y_train_res = smt.fit_resample(X_train, y_train)

print("After applying SMOTE-Tomek on Imbalanced Classification: ")
print(y_train_res.value_counts())

# distribution of imbalanced classification after applying resampling technique
y_train_res.value_counts().plot(kind = 'barh', color = ['skyblue', 'limegreen'])
plt.title("Distribution of Imbalanced Classification (After Applying SMOTE-Tomek)", fontweight = 'bold', fontsize = 15)
plt.xlabel("Number of Observation", fontweight = 'bold', fontsize = 12)
plt.ylabel("Class", fontweight = 'bold', fontsize = 12)
plt.show()


###################################################################################
################### 9. Build Adaboost Classifier (Baseline)  ######################
###################################################################################

# define the adaboost classifier params 
ada_model = AdaBoostClassifier(n_estimators = 250, 
                               learning_rate = 0.1,
                               random_state = 42)
ada_model.fit(x_train_res, y_train_res)


#######################################################################################
################### 10. Evaluate Adaboost Classifier (Baseline)  ######################
#######################################################################################
# prediction on training set
y_pred_train = ada_model.predict(x_train_res)

# prediction on testing set
y_pred_test = ada_model.predict(X_test)

# classification report on training set 
print(classification_report(y_train_res, y_pred_train, target_names = ['Non-Diabetes', 'Diabetes']))

# classification report on testing set
print(classification_report(y_test, y_pred_test, target_names = ['Non-Diabetes', 'Diabetes']))

# evaluation metrics on training set
print()
print("Training Set: ")
print("Accuracy Score: ", accuracy_score(y_train_res, y_pred_train))
print("Precision Score: ", precision_score(y_train_res, y_pred_train))
print("F1 Score: ", f1_score(y_train_res, y_pred_train))
print("Recall Score: ", recall_score(y_train_res, y_pred_train))
print("ROC-AUC Score: ", roc_auc_score(y_train_res, y_pred_train))

# evaluation metrics on testing set
print()
print("Testing Set: ")
print("Accuracy Score: ", accuracy_score(y_test, y_pred_test))
print("Precision Score: ", precision_score(y_test, y_pred_test))
print("F1 Score: ", f1_score(y_test, y_pred_test))
print("Recall Score: ", recall_score(y_test, y_pred_test))
print("ROC-AUC Score: ", roc_auc_score(y_test, y_pred_test))

# confusion matrix on training set
cnf_matrix_train = confusion_matrix(y_train_res, y_pred_train)
sns.heatmap(cnf_matrix_train, annot = True, cmap = 'coolwarm')
plt.title("Confusion Matrix on Training Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("Predicted Label", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Label", fontweight = 'bold', fontsize = 12)
plt.show()

# confusion matrix on testing set
cnf_matrix_train = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cnf_matrix_train, annot = True, cmap = 'coolwarm')
plt.title("Confusion Matrix on Testing Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("Predicted Label", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Label", fontweight = 'bold', fontsize = 12)
plt.show()

# roc-auc-curve on training set
train_probs = ada_model.predict_proba(x_train_res)[:, -1]
fpr_train, tpr_train, threshold_train = roc_curve(y_train_res, train_probs, pos_label = 1)

sns.lineplot(x = fpr_train, y = tpr_train, color = 'crimson', label = "Adaboost")
sns.lineplot(x = [0, 1], y = [0, 1], label = "No-Skill")
plt.title("ROC-Curve on Training Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("False Positive Rate (FPR)", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Positive Rate (TPR)", fontweight = 'bold', fontsize = 12)
plt.legend(shadow = True)
plt.show()

# roc-auc-curve on testing set
test_probs = ada_model.predict_proba(X_test)[:, -1]
fpr_test, tpr_test, threshold_train = roc_curve(y_test, test_probs, pos_label = 1)

sns.lineplot(x = fpr_test, y = tpr_test, color = 'crimson', label = "Adaboost")
sns.lineplot(x = [0, 1], y = [0, 1], label = "No-Skill")
plt.title("ROC-Curve on Testing Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("False Positive Rate (FPR)", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Positive Rate (TPR)", fontweight = 'bold', fontsize = 12)
plt.legend(shadow = True)
plt.show()


#############################################################################################
################### 11. Hyperparameters Tuning on Adaboost Classifier  ######################
#############################################################################################
#define grid values to search 
params = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
          'learning_rate': [0.001, 0.01, 0.1, 1.0]}

# define the adaboost classifier
ada_model = AdaBoostClassifier(random_state = 42)

# define evaluation procedure
cv = ShuffleSplit(train_size = 0.8, test_size = 0.2, n_splits = 5, random_state = 42)

# define the grid search procedure
grid_search = GridSearchCV(estimator = ada_model, 
                           param_grid = params, 
                           n_jobs = 1, 
                           verbose = 3, 
                           cv = cv,
                           scoring = 'accuracy', 
                           return_train_score = True)

gs_result = grid_search.fit(x_train_res, y_train_res)

print("Best Hyperparameters: ", gs_result.best_params_)
print("Best Score: ", gs_result.best_score_)

# store the best hyperparameters into dataframe
best_hyperparameters = pd.Series(data = gs_result.best_params_)
print(best_hyperparameters)

# summarize all scores that were evaluated
means = gs_result.cv_results_['mean_test_score']
stds = gs_result.cv_results_['std_test_score']
params = gs_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("{:.3f} ({:.3f}) with : {}".format(mean, stdev, param))

#############################################################################################
################### 12. Rebuild Adaboost Classifier with Best Hyperparameters  ##############
#############################################################################################
ada_model2 = AdaBoostClassifier(n_estimators = int(best_hyperparameters['n_estimators']),
                                learning_rate = best_hyperparameters['learning_rate'])
ada_model2.fit(x_train_res, y_train_res)

#############################################################################################
################### 13. Evaluate Performance on Rebuild Adaboost Classifier  ################
#############################################################################################

# prediction on training set
y_pred_train2 = ada_model2.predict(x_train_res)

# prediction on testing set
y_pred_test2 = ada_model2.predict(X_test)

# classification report on training set 
print(classification_report(y_train_res, y_pred_train2, target_names = ['Non-Diabetes', 'Diabetes']))

# classification report on testing set
print(classification_report(y_test, y_pred_test2, target_names = ['Non-Diabetes', 'Diabetes']))

# evaluation metrics on training set
print()
print("Training Set: ")
print("Accuracy Score: ", accuracy_score(y_train_res, y_pred_train2))
print("Precision Score: ", precision_score(y_train_res, y_pred_train2))
print("F1 Score: ", f1_score(y_train_res, y_pred_train2))
print("Recall Score: ", recall_score(y_train_res, y_pred_train2))
print("ROC-AUC Score: ", roc_auc_score(y_train_res, y_pred_train2))

# evaluation metrics on testing set
print()
print("Testing Set: ")
print("Accuracy Score: ", accuracy_score(y_test, y_pred_test2))
print("Precision Score: ", precision_score(y_test, y_pred_test2))
print("F1 Score: ", f1_score(y_test, y_pred_test2))
print("Recall Score: ", recall_score(y_test, y_pred_test2))
print("ROC-AUC Score: ", roc_auc_score(y_test, y_pred_test2))

# confusion matrix on training set
cnf_matrix_train = confusion_matrix(y_train_res, y_pred_train2)
sns.heatmap(cnf_matrix_train, annot = True, cmap = 'coolwarm')
plt.title("Confusion Matrix on Training Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("Predicted Label", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Label", fontweight = 'bold', fontsize = 12)
plt.show()

# confusion matrix on testing set
cnf_matrix_train = confusion_matrix(y_test, y_pred_test2)
sns.heatmap(cnf_matrix_train, annot = True, cmap = 'coolwarm')
plt.title("Confusion Matrix on Testing Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("Predicted Label", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Label", fontweight = 'bold', fontsize = 12)
plt.show()

# roc-auc-curve on training set
train_probs2 = ada_model2.predict_proba(x_train_res)[:, -1]
fpr_train, tpr_train, threshold_train = roc_curve(y_train_res, train_probs2, pos_label = 1)

sns.lineplot(x = fpr_train, y = tpr_train, color = 'crimson', label = "Adaboost")
sns.lineplot(x = [0, 1], y = [0, 1], label = "No-Skill")
plt.title("ROC-Curve on Training Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("False Positive Rate (FPR)", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Positive Rate (TPR)", fontweight = 'bold', fontsize = 12)
plt.legend(shadow = True)
plt.show()

# roc-auc-curve on testing set
test_probs2 = ada_model2.predict_proba(X_test)[:, -1]
fpr_test, tpr_test, threshold_train = roc_curve(y_test, test_probs2, pos_label = 1)

sns.lineplot(x = fpr_test, y = tpr_test, color = 'crimson', label = "Adaboost")
sns.lineplot(x = [0, 1], y = [0, 1], label = "No-Skill")
plt.title("ROC-Curve on Testing Set", fontweight = 'bold', fontsize = 15)
plt.xlabel("False Positive Rate (FPR)", fontweight = 'bold', fontsize = 12)
plt.ylabel("True Positive Rate (TPR)", fontweight = 'bold', fontsize = 12)
plt.legend(shadow = True)
plt.show()

########################################################
################### 14. Save the Model  ################
########################################################
import pickle

filepath_ = r"C:\Users\jeffr\Desktop\diabetes prediction\adaBoost_model.sav"
pickle.dump(ada_model2, open(filepath_, 'wb'))

