import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, confusion_matrix


def remove_dummification_duplicates(X, features, categorical_columns, expanded_features):
    """
    Removes extra column after categorical variables have been dummified
    :param X: a dataframe of training variables
    :param features: a list of features to extract from the dataframe
    :param categorical_columns: a list of variables from X that are categorical
    :param expanded_features: the expanded categorical columns
    :return: all selected features, with one categorical column dropped for each categorical variable after
    dummification
    """
    for feature in features:
        # names i.e. with the extension (dummification), then removal of the first column
        if feature in categorical_columns:
            # drop the feature from the list of expanded features
            expanded_features.remove(feature)
            unique_feature_values = X[feature].unique().tolist()
            # include in expanded features the new features in the format: feature+value
            for value in sorted(unique_feature_values)[1:]:  # ignoring the first value because it'd serve as reference
                expanded_features.append(feature + "_" + str(value))
    return expanded_features


def func(features, outcome, outcome_variables, k_neighbors=5, oversample=False, num_of_features=10,
         predict=False, use_dosage_vals=True, dosage_cols=None, code_with_neg_one=True,
         all_data=False, save_model=False, save_file_name=None, must_include_features=None):
    """
    Function to fit logistic regression model, following recursive feature elimination (RFE) on training data to
    obtain association between variables and outcomes
    :param features: independent variables
    :param outcome: dependent variable
    :param outcome_variables: a list of dependent variables (these are passed so that they can be removed from the
    dataframe before the model is fit)
    :param k_neighbors: number of neighbors - required for SMOTE (oversampling)
    :param oversample: boolean value to check if data should be oversampled
    :param num_of_features: number of features to select using RFE
    :param predict: boolean value on whether after the model is trained, it should be evaluated by generating
    predictions from a "test-set"
    :param use_dosage_vals: checks whether the boolean variables on whether a patient received a specific drug each of
    the admission first 5 admission days should be included in the learned model
    :param dosage_cols: if yes, (to use_dosage_vals above), the dosage columns are passed here as a list
    :param code_with_neg_one: checks whether instead of using (0,1) values for the boolean variables,
    (-1, 1) should be used instead. Sometimes, multiple multiplications with 0 can produce errors/and instabilities,
    hence the choice to code values as 1 (positive), and -1 (negative).
    :param all_data: boolean function to check whether all the data should be used to learn the model,
    or if part of it should be used for train and part for test
    :param save_model: checks if the trained model should be saved
    :param save_file_name: if model is to be saved, save_file_name is the path and filename it will be saved under
    :param must_include_features: features that must be included in the model.
    whether selected by RFE or not, these features would be in the model
    :return: results of the fit model
    """
    categorical_columns = ["ethnicity", "insurance", "gender", "icd9_code", "discharge_location", "white?",
                           "white?_edited"]
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    X = df.drop(outcome, axis=1)
    y = df[outcome]

    # dummify all categorical columns in X
    for col in X.columns:
        if col in categorical_columns:
            cat_features_list = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = X.join(cat_features_list)

    expanded_features = list(set(features))
    expanded_features = remove_dummification_duplicates(X=X, features=features,
                                                        categorical_columns=categorical_columns,
                                                        expanded_features=expanded_features)
    X_cols = X.columns.values.tolist()
    features_to_keep = [i for i in X_cols if i not in categorical_columns]

    X_final = X[features_to_keep]

    x_final_cols = X_final.columns

    if code_with_neg_one:
        for col in x_final_cols:
            if col not in outcome_variables:
                if sorted(list(set(X_final[col].values))) == [0, 1]:  # check if the column only contains 0,1 values
                    X_final[col] = X_final[col].mask(X_final[col] == 0).fillna(
                        -1)  # note: throws warning. Use .iloc instead
    # Replace NA values in dosage columns with 0
    if use_dosage_vals:
        if dosage_cols is not None:
            common_cols = list(
                set(X_final.columns).intersection(dosage_cols))
            X_final[common_cols] = X_final[common_cols].fillna(0)
        else:
            raise ValueError("Error! To analyze the data using dosage values, "
                             "you need to specify which columns represent dosage values")

    if all_data:
        X_train_full = X_final
        y_train = y
        X_test_full = X_final
        y_test = y
    else:
        X_train_full, X_test_full, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=0)
    x_train_cols = X_train_full.columns

    # impute missing values for numerical features
    if X_train_full.isnull().values.any():
        # print("found a numeric column with nulls named ", col)
        for col in x_train_cols:
            if X_train_full[col].isnull().values.any():
                count = X_train_full[col].isna().sum()
                length = len(X_train_full[col])

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

    for col in cols_with_nans:
        X_train_col_mean = X_train_full[col].mean()
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

    # Using RFE to select features that are predictive of other outcome:
    lr = LogisticRegression(max_iter=50000)
    if must_include_features is not None:
        num_of_features = num_of_features - len(must_include_features)
    rfe = RFE(lr, n_features_to_select=num_of_features)
    rfe = rfe.fit(X_train, y_train.values.ravel())

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
    important_features = list(set(important_features))
    model = sm.Logit(endog=y_train.values.ravel(), exog=X_train[important_features]).fit()
    print("model fit completed. fit coefs and errors ==> ", model.summary())
    if save_model:
        with open("./data/results/outcomes/model-fits/lr-rfe-nuf-" + str(num_of_features)
                  + "-model-fit-extras-" + save_file_name + ".txt", "w") as f:
            f.write(model.summary().as_text())

    # predict on test set and obtain model statistics
    if predict:
        pred_probs = model.predict(X_test[important_features])
        pred = round(pred_probs)
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

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true=list(y_test.values.ravel()), y_score=list(pred_probs))

        import json
        obj = {"acc": acc, "auc": auc, "conf-matrix": conf_mat.tolist(), "coefs": coefs.to_json(),
               "p-values": pvalues.to_json(),
               "prsquared": prsquared,
               "llf": llf, "llr": llr, "llr-pvalue": llr_pvalue, "confidence-intervals": conf_intervals.to_json()}

        with open("./data/results/outcomes/predictions/method-lr-ref-nuf-" + str(
                num_of_features) + "-model-props-and-predictions.text", 'w') as f:
            json.dump(obj, f, indent=2)

        # fitted model accuracy
        return {"X_train": X_train, "X_test": X_test_full, "y_test": y_test}, list(
            pred), conf_mat, acc, coefs, llf, llr, llr_pvalue, prsquared, pvalues, conf_intervals, auc


def process_model_fit_output(model_variables):
    """
    Runs func(), then processes results of the trained model. Importantly, it reads the coefficients and
    computes odds ratios
    :param model_variables: an object of parameters required by func().
    :return: nothing. results are written to a file in ./data/results/outcomes/
    """
    nuf = model_variables["num_of_features"]
    save_name = model_variables["save_file_name"]
    data, *rest = func(**model_variables)

    coefs_all = rest[3]

    variables = coefs_all.keys().tolist()
    coefs_values = coefs_all.values.tolist()
    prsquared = rest[7]
    pvalues = rest[8].values.round(decimals=5).tolist()

    ci_intervals = rest[9].values.tolist()
    or_ci_intervals = np.exp(ci_intervals)

    odds_ratio = np.exp(coefs_values)

    df = pd.DataFrame()
    df["variable"] = variables
    df["odds-ratio"] = [round(num, 3) for num in odds_ratio]
    df["coef"] = [round(num, 3) for num in coefs_values]
    df["p-value"] = pvalues
    df["or ci low"] = [round(num, 3) for num in or_ci_intervals[:, 0]]
    df["or ci high"] = [round(num, 3) for num in or_ci_intervals[:, 1]]
    df.to_csv("./data/results/outcomes/summarized-model-fits/nuf-" + str(nuf) + "-extras-" + save_name + ".csv")


def main():
    """
    Control logic to run the entire pipeline.
    Trains the model, and analyzes it's outcome, and everything is saved in ./data/results/outcomes/
    :return:
    """
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
    musts_with_age = ["admit_duration(days)", "non_stemi?", "c-shock?", "surgical_procedure?", "gender_m",
                      "white?_edited_yes", "white?_edited_no", "age"]

    obj = {"features": features_with_dosages,
           "outcome": "discharge-to-home?",
           "outcome_variables": outcomes,
           "dosage_cols": dosage_columns,
           "num_of_features": 19,
           "predict": True, "all_data": True, "save_model": True,
           "save_file_name": "all-features-with-musts", "must_include_features": musts}
    obj_with_age = {"features": features_with_dosages,
                    "outcome": "discharge-to-home?",
                    "outcome_variables": outcomes,
                    "dosage_cols": dosage_columns,
                    "num_of_features": 19,
                    "predict": True, "all_data": True, "save_model": True,
                    "save_file_name": "all-features-with-musts-and-age",
                    "must_include_features": musts_with_age}
    process_model_fit_output(model_variables=obj)
    process_model_fit_output(model_variables=obj_with_age)


if __name__ == '__main__':
    main()
