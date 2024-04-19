import pandas as pd
import numpy as np
from collections import Counter
import scipy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
# from pandas.api.types import is_numeric_dtype
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from time import time
import matplotlib.pyplot as plt
from sklearn.svm import l1_min_c
from sklearn.metrics import accuracy_score, confusion_matrix


def expand_features_list(X, features, categorical_columns, expanded_features):
    """

    :param X:
    :param features:
    :param categorical_columns:
    :param expanded_features:
    :return:
    """
    for feature in features:
        # names i.e. with the extension (dummification), then removal of the first column
        if feature in categorical_columns:
            # drop the feature from the list of expanded features
            expanded_features.remove(feature)
            unique_feature_values = X[feature].unique().tolist()
            print("feature = ", feature, " has unique values : ", unique_feature_values)
            # include in expanded features the new features in the format: feature+value
            for value in sorted(unique_feature_values)[1:]:  # ignoring the first value because it'd serve as reference
                expanded_features.append(feature + "_" + str(value))
    return expanded_features


def func(features, outcome, outcome_variables, k_neighbors=5, oversample=False, num_of_features=10,
         predict=False, use_dosage_vals=True, dosage_cols=None, code_with_neg_one=True,
         all_data=False, save_model=False, save_file_name=None, must_include_features=None):
    """

    :param features:
    :param outcome:
    :param outcome_variables:
    :param k_neighbors:
    :param oversample:
    :param num_of_features:
    :param predict:
    :param use_dosage_vals:
    :param dosage_cols:
    :param code_with_neg_one:
    :param all_data:
    :param save_model:
    :param save_file_name:
    :param must_include_features:
    :return:
    """
    categorical_columns = ["ethnicity", "insurance", "gender", "icd9_code", "discharge_location", "white?",
                           "white?_edited"]
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    print("df columns = ", df.columns, len(df.columns))
    X = df.drop(outcome, axis=1)
    print("X column names = ", X.columns, len(X.columns))
    y = df[outcome]

    # dummify all categorical columns in X
    for col in X.columns:
        if col in categorical_columns:
            cat_features_list = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = X.join(cat_features_list)

    expanded_features = list(set(features))
    print("expanded features 1 = ", expanded_features, " with len = ", len(expanded_features),
          " compared to original features list with len = ", len(features))

    expanded_features = expand_features_list(X=X, features=features, categorical_columns=categorical_columns,
                                             expanded_features=expanded_features)

    print("expanded features 2 = ", expanded_features, " with len = ", len(expanded_features),
          " compared to original features list with len = ", len(features))

    X_cols = X.columns.values.tolist()
    print("X cols = ", list(X_cols))
    features_to_keep = [i for i in X_cols if i not in categorical_columns]

    X_final = X[features_to_keep]
    # print("X_final contains = ", X_final)
    print("X's columns names = ", X_final.columns.values)  # "\n", X_final.head())

    x_final_cols = X_final.columns

    if code_with_neg_one:
        for col in x_final_cols:
            if col not in outcome_variables:
                if sorted(list(set(X_final[col].values))) == [0, 1]:  # check if the column only contains 0,1 values
                    print("X_final[col] before = ", col, list(X_final[col]))
                    X_final[col] = X_final[col].mask(X_final[col] == 0).fillna(
                        -1)  # note: throws warning. Use .iloc instead
                    print("X_final[col] after = ", col, list(X_final[col]))
    # Replace NA values in dosage columns with 0
    if use_dosage_vals:
        if dosage_cols is not None:
            common_cols = list(
                set(X_final.columns).intersection(dosage_cols))  # TODO: think about why you're doing this
            X_final[common_cols] = X_final[common_cols].fillna(0)
        else:
            print("Error! To analyze the data using dosage values, "
                  "you need to specify which columns represent dosage values")

    if all_data:
        X_train_full = X_final
        y_train = y
        X_test_full = X_final
        y_test = y
        print("len of dataset = ", len(X_train_full), len(y_train))
    else:
        X_train_full, X_test_full, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=0)
    print("length of X-train = ", len(X_train_full), " and that of X-test = ", len(X_test_full), "and that of y_train "
                                                                                                 "= ",
          len(y_train), " and that of y-test = ", len(y_test))
    x_train_cols = X_train_full.columns

    # impute missing values for numerical features
    if X_train_full.isnull().values.any():
        # print("found a numeric column with nulls named ", col)
        for col in x_train_cols:
            if X_train_full[col].isnull().values.any():
                count = X_train_full[col].isna().sum()
                length = len(X_train_full[col])
                print("col = ", col, " has ", count, " out of ", length,
                      "null values, which is ", float(count / length) * 100, "%")

        imp = IterativeImputer(max_iter=500, random_state=0)
        imp.fit(X_train_full)
        IterativeImputer(random_state=0)
        X_train_mod = imp.transform(X_train_full)
        X_train_full = pd.DataFrame(data=X_train_mod, columns=x_train_cols)

    # impute test set
    # get columns with NaNs so we know which ones we'll impute
    X_test_cols = X_test_full.columns

    cols_with_nans = []
    for col in X_test_cols:
        if X_test_full[col].isnull().values.any():
            cols_with_nans.append(col)
    print("X-test columns with nulls values = ", cols_with_nans)

    for col in cols_with_nans:
        X_train_col_mean = X_train_full[col].mean()
        print("X_test col  = ", col, " and X_train mean for the col = ", X_train_col_mean)
        X_test_full[col] = X_test_full[col].fillna(X_train_col_mean)  # note: this is also throwing a warning requiring
        # the use loc

    X_train = X_train_full[expanded_features]
    X_test = X_test_full[expanded_features]

    # oversampling the training data to accommodate the class imbalance
    if oversample:
        os = SMOTE(random_state=0, k_neighbors=k_neighbors)
        os_X, os_y = os.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(data=os_X, columns=x_train_cols)
        y_train = pd.DataFrame(data=os_y, columns=[outcome])

        print("length of oversampled data is ", len(X_train))
        print("Number of negative-class-label in oversampled data", len(y_train[y_train[outcome] == 0]))
        print("Number of positive-class-label in oversampled data", len(y_train[y_train[outcome] == 1]))
        print("Proportion of negative-class-label data in oversampled data is ",
              len(y_train[y_train[outcome] == 0]) / len(os_X))
        print("Proportion of positive-class0label data in oversampled data is ",
              len(y_train[y_train[outcome] == 1]) / len(os_X))

    # model = None
    # important_features = None

    # Using RFE to select features that are predictive of other outcome:
    lr = LogisticRegression(max_iter=50000)
    if must_include_features is not None:
        num_of_features = num_of_features - len(must_include_features)
    print("num of features = ", num_of_features)
    rfe = RFE(lr, n_features_to_select=num_of_features)
    rfe = rfe.fit(X_train, y_train.values.ravel())
    print(rfe.support_, len(rfe.support_))
    print(rfe.ranking_, len(rfe.ranking_))
    print("Xtrain cols = ", X_train.columns, len(X_train.columns))

    # pick the predictive features
    ranks = rfe.ranking_
    cols = X_train.columns.values.tolist()
    if must_include_features is not None:
        important_features = must_include_features
    else:
        important_features = []
    for i in range(len(cols)):
        if ranks[i] == 1:
            important_features.append(cols[i])
    print("important features are: ", important_features, len(important_features))
    important_features = list(set(important_features))
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train[important_features]).fit()
    print(model.summary())
    if save_model:
        with open("./data/results/outcomes/model-fits/lr-rfe-nuf-" + str(num_of_features)
                  + "-model-fit-extras-" + save_file_name + ".txt", "w") as f:
            f.write(model.summary().as_text())

    # predict on test set and obtain model statistics
    if predict:
        # llf = None
        # llr = None
        # llr_pvalue = None
        # prsquared = None
        # coefs = None
        # pvalues = None
        # conf_intervals = None

        pred_probs = model.predict(X_test[important_features])
        print("pred before rounding = ", pred_probs, type(pred_probs))
        pred = round(pred_probs)
        print("pred after rounding = ", pred)
        llf = model.llf
        llr = model.llr
        llr_pvalue = model.llr_pvalue
        prsquared = model.prsquared
        coefs = model.params
        pvalues = model.pvalues
        conf_intervals = model.conf_int(alpha=0.05)  # 95% confidence level

        # get confusion matrix and accuracy of the prediction
        conf_mat = confusion_matrix(y_true=list(y_test.values.ravel()), y_pred=pred)
        acc = accuracy_score(y_true=list(y_test.values.ravel()), y_pred=pred)
        print("confusion matrix = ", conf_mat)

        print("type of coefs = ", type(coefs), " and type of p-values = ", type(pvalues), " and type of llf = ",
              type(llf),
              " and type of acc = ", type(acc), type(conf_mat))

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true=list(y_test.values.ravel()), y_score=list(pred_probs))
        print("auc = ", auc)

        import json
        obj = {"acc": acc, "auc": auc, "conf-matrix": conf_mat.tolist(), "coefs": coefs.to_json(),
               "p-values": pvalues.to_json(),
               "prsquared": prsquared,
               "llf": llf, "llr": llr, "llr-pvalue": llr_pvalue, "confidence-intervals": conf_intervals.to_json()}

        with open("./data/results/outcomes/predictions/method-lr-ref-nuf-" + str(
                num_of_features) + "-model-props-and-predictions.text", 'w') as f:
            json.dump(obj, f, indent=2)

        # with open("./data/predictions/method-" + method + "-nuf-" + str(
        #         num_of_features) + "-model-props-and-predictions.text", 'w') as f:
        #     f.write(json.dumps(obj))

        # fitted model accuracy
        print("accuracy = ", acc)
        return {"X_train": X_train, "X_test": X_test_full, "y_test": y_test}, list(
            pred), conf_mat, acc, coefs, llf, llr, llr_pvalue, prsquared, pvalues, conf_intervals, auc


def process_model_fit_output(model_variables):
    nuf = model_variables["num_of_features"]
    save_name = model_variables["save_file_name"]
    data, *rest = func(**model_variables)

    # preds, conf_mat, acc, coefs, llf, llr, llr_pvalue, prsquared, pvalues, conf_intervals, auc

    coefs_all = rest[3]
    # print("coefs all = ", coefs_all.sort_index())

    variables = coefs_all.keys().tolist()
    print("variables = ", variables)
    coefs_values = coefs_all.values.tolist()
    print("coefs values = ", coefs_values)
    prsquared = rest[7]
    pvalues = rest[8].values.round(decimals=5).tolist()

    ci_intervals = rest[9].values.tolist()
    print("ci intervals = ", ci_intervals)
    or_ci_intervals = np.exp(ci_intervals)
    print("or ci intervals = ", or_ci_intervals)

    odds_ratio = np.exp(coefs_values)
    print("odds_ratio = ", odds_ratio)

    df = pd.DataFrame()
    df["variable"] = variables
    df["odds-ratio"] = [round(num, 3) for num in odds_ratio]
    df["coef"] = [round(num, 3) for num in coefs_values]
    df["p-value"] = pvalues
    df["or ci low"] = [round(num, 3) for num in or_ci_intervals[:, 0]]
    df["or ci high"] = [round(num, 3) for num in or_ci_intervals[:, 1]]

    print("df head = ", df.head())
    df.to_csv("./data/results/outcomes/summarized-model-fits/nuf-" + str(nuf) + "-extras-" + save_name + ".csv")


def main():
    features_with_dosages = ["age", "admit_duration(days)", "non_stemi?", "shock?", "c-shock?", "aspirin?",
                             "received_statin?", "received_anti-platelet?", "received_ace-inhibitor?",
                             "received_beta-blocker?",
                             "surgical_procedure?", "liver_conditions?", "kidney_conditions?", "received_pain_med?",
                             "received_narcotic?", "received_procedure?", "non_surgical_procedure?", "other_procedure?",
                             "d1_mme", "ad-average-mme", "d1_nnd", "ad-average-nnd", "d1_ad", "ad-average-asd",
                             "d1_std", "ad-average-statin", "d1_and", "ad-average-anti-platelet", "d1_acd",
                             "ad-average-ace-inhibitor", "d1_bed", "ad-average-beta-blocker",
                             "received_narc_only_day1?", "received_n_narc_only_day1?",
                             "received_combined_therapy_day1?", "initial_trop_T", "peak_trop_T",
                             "initial_c", "gender", "white?", "white?_edited", "insurance", "white?_yes_f",
                             "white?_yes_m",
                             "white?_unknown_f", "white?_unknown_m", "white?_no_f", "white?_no_m",
                             "anterior_lateral_mi?",
                             "inferior_posterior_mi?", "other_mi?"]
    dosage_columns = ["d1_mme", "d2_mme", "d3_mme", "d4_mme", "d5_mme", "d1_nnd", "d2_nnd", "d3_nnd", "d4_nnd",
                      "d5_nnd", "d1_ad", "d2_ad", "d3_ad", "d4_ad", "d5_ad", "d1_std", "d2_std", "d3_std", "d4_std",
                      "d5_std", "d1_and", "d2_and", "d3_and", "d4_and", "d5_and", "d1_acd", "d2_acd", "d3_acd",
                      "d4_acd", "d5_acd", "d1_bed", "d2_bed", "d3_bed", "d4_bed", "d5_bed", "ad-total-mme",
                      "ad-average-mme",
                      "ad-total-orders", "ad-average-nnd", "aspirin-total-dose", "ad-average-asd", "statin-orders",
                      "ad-average-statin", "anti-platelet-orders", "ad-average-anti-platelet", "ace-inhibitor-orders",
                      "ad-average-ace-inhibitor", "beta-blocker-orders", "ad-average-beta-blocker"]
    outcomes = ["discharge_location", "died?", "discharge-to-home?"]
    # social_demographic_features = ["gender", "white?", "white?edited", "insurance", "white?_yes_f", "white?_yes_m",
    #                                "white?_unknown_f",
    #                                "white?_unknown_m", "white?_no_f", "white?_no_m"]

    # pain_medication_features = ["received_pain_med?",
    #                             "received_narcotic?",
    #                             "d1_mme", "ad-average-mme", "d1_nnd", "ad-average-nnd",
    #                             "received_narc_only_day1?", "received_n_narc_only_day1?",
    #                             "received_combined_therapy_day1?"]
    # ami_medication_features = ["aspirin?", "received_statin?", "received_anti-platelet?", "received_ace-inhibitor?",
    #                            "received_beta-blocker?", "d1_ad", "ad-average-asd",
    #                            "d1_std", "ad-average-statin", "d1_and", "ad-average-anti-platelet", "d1_acd",
    #                            "ad-average-ace-inhibitor", "d1_bed", "ad-average-beta-blocker"]

    # fit and evaluate model with certain features fixed
    musts = ["admit_duration(days)", "non_stemi?", "c-shock?", "surgical_procedure?", "gender_m",
             "white?_edited_yes", "white?_edited_no"]

    obj = {"features": features_with_dosages,
           "outcome": "discharge-to-home?",
           "outcome_variables": outcomes,
           "dosage_cols": dosage_columns,
           "num_of_features": 19,
           "predict": True, "all_data": True, "save_model": True,
           "save_file_name": "all-features-with-musts", "must_include_features": musts}
    process_model_fit_output(model_variables=obj)


if __name__ == '__main__':
    main()
