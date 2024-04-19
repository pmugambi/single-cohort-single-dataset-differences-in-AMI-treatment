from __future__ import division
import datetime

from datetime import date, timedelta

import pandas as pd
from collections import Counter
import researchpy as rp
from scipy import stats
from patsy import dmatrices
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import math

from data_processing import assign_mme_values, compute_dosage_per_day, sum_mme_per_day, mme_per_admit_day, \
    pad_list_with_zeros, assign_medication_types, sum_non_narcs_per_day, sum_aspirin_per_day


def read_results(filename):
    data = pd.read_csv(filename)
    hadm_ids = data["hadm_id"].tolist()
    return hadm_ids


def read_first_prescriptions(filename):
    df = pd.read_csv(filename)
    df = df.sort_values(['HADM_ID', 'startdate'])
    # df_new = df.ffill().drop_duplicates('HADM_ID', keep='first')

    narcotic_check = []
    drugs = df['drug']
    for drug in drugs:
        if "fentanyl citrate" in drug.lower() or "hydromorphone" in drug.lower() or "morphine" in drug.lower():
            narcotic = "yes"
        else:
            narcotic = "no"
        narcotic_check.append(narcotic)
    df["is_narcotic"] = narcotic_check
    df.to_csv(filename)
    return df


def compute_chi_square(filename, outcome_var, sensitive_var):
    df = pd.read_csv(filename)
    # print(rp.summary_cat(df[[outcome_var, sensitive_var]]))

    table, results = rp.crosstab(df[outcome_var], df[sensitive_var], prop='col',
                                 test='chi-square')
    # print(table)
    print("rp results = ", results)
    crosstab = pd.crosstab(df[outcome_var], df[sensitive_var])
    print("pd crosstab = ", crosstab)
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    print("chi2, p_value, dof from pd.chi = ", chi2, p, dof)
    print("len of df being investigated = ", len(df))
    return results


def assign_dosage_days(filename):
    data = pd.read_csv(filename)
    data = data[pd.notnull(data['dose_val_rx'])]
    dose_amounts = list(data["dose_val_rx"])
    drugs = list(data["drug"])
    print("drugs = ", list(drugs))
    dose_units = list(data["dose_unit_rx"])
    # print("dose units = ", list(dose_units))
    genders = list(data["gender"])
    routes = list(data["route"])

    days_list = []
    gender_nums = []

    # print(data.head())

    print(drugs[142])

    for i in range(0, len(drugs)):
        drug = drugs[i].lower()
        print("drug = ", i, drug)
        dose_amount = dose_amounts[i]
        # print("fails here = ", dose_units[i])
        dose_unit = dose_units[i].lower()
        gender = genders[i].lower()
        route = routes[i].lower()
        # print("fails here now = ", dose_amount)
        if '-' in dose_amount:
            dose_amount = dose_amount.split("-")[0]
        dose_amount = float(dose_amount)

        if drug == "fentanyl citrate" and dose_unit == "mcg" and route == "iv":
            # days = dose_amount / float(12.5)
            days = dose_amount / float(25)
        elif drug == "fentanyl citrate" and dose_unit == "mcg" and route == "iv drip":
            days = dose_amount / float(2500)
        elif drug == "fentanyl citrate" and dose_unit == "mg" and route == "iv drip":
            days = dose_amount / float(2.5)
        elif drug == "fentanyl citrate" and dose_unit == "mcg" and route == "pb":
            days = dose_amount / float(2500)
        elif drug == "fentanyl citrate" and dose_unit == "mcg":
            days = dose_amount / float(25)
        # elif drug == "fentanyl citrate" and dose_unit == "mg":
        #     days = (dose_amount * 1000) / float(25)
        elif drug == "morphine sulfate" and dose_unit == "mg" and route == "iv drip":
            days = dose_amount / float(10)
        elif drug == "morphine sulfate" and dose_unit == "mg":
            days = dose_amount
        elif "morphine sr" in drug and dose_unit == "mg":
            days = dose_amount / float(30)
        elif "hydromorphone" in drug and dose_unit == "mg" and route == "iv":
            days = dose_amount / float(2)
        elif "hydromorphone" in drug and dose_unit == "mg" and route == "po":
            days = dose_amount / float(2)
        elif "hydromorphone" in drug and dose_unit == "mg" and route == "sc":
            days = dose_amount / float(0.5)
        elif "hydromorphone" in drug and dose_unit == "mg":
            days = dose_amount / float(0.5)
        elif drug == "morphine sulfate ir" and dose_unit == "mg":
            days = dose_amount / float(15)
        elif drug == "morphine sulfate sr" and dose_unit == "mg":
            days = dose_amount / float(15)
        elif drug == "acetaminophen" and dose_unit == "mg":
            days = dose_amount / float(4000)
        elif drug == "ibuprofen" and dose_unit == "mg":
            days = dose_amount / float(1200)
        elif drug == "asprin" and dose_unit == "mg":
            days = dose_amount / float(4000)
        else:
            days = -1
        days_list.append(days)

        if gender == "m":
            gender_num = 0
        else:
            gender_num = 1
        gender_nums.append(gender_num)

    data.insert(3, "dosage_days", days_list)
    # data["dosage_days"] = days_list
    data["gender_num"] = gender_nums
    print(data.head())
    data.to_csv("./data/prescriptions_with_dosage_days_computed.csv")
    return data


def sum_days(filename):
    data = pd.read_csv(filename)
    d = data.groupby("HADM_ID")
    groups = d.groups
    g_keys = list(groups.keys())

    print(data.head())

    df = pd.DataFrame()
    for group in g_keys:
        f = d.get_group(group)
        f_prime = f.head(1)
        days = sum(f["dosage_days"])
        narcotics = f.where(f["is_narcotic"] == "yes")
        non_narcotics = f.where(f["is_narcotic"] == "no")
        narcotic_days = sum(narcotics["dosage_days"])
        non_narcotics_days = sum(non_narcotics["dosage_days"])
        # print("days = ", days, " for group = ", group)
        f_prime.drop("dosage_days", axis=1)
        f_prime.insert(7, "sum_dosage_days", days)
        f_prime.insert(8, "sum_narcotics_dosage_days", narcotic_days)
        f_prime.insert(9, "sum_non_narcotics_dosage_days", non_narcotics_days)
        gender = list(f_prime["gender"])[0]
        if gender == "M":
            gender_num = 0
        else:
            gender_num = 1
        f_prime["gender_num"] = gender_num
        f_p = f_prime[7:]
        # print("f_p = ", f_p)
        df = pd.concat([df, f_prime.iloc[:, 7:]])
    print(df.head())
    df.to_csv("./data/prescriptions_with_summed_days.csv")


def compute_point_biserial_corr(filename, dependent_var, independent_var):
    data = pd.read_csv(filename)
    print(type(data[independent_var]))
    corr = stats.pointbiserialr(data[independent_var], data[dependent_var])
    print(corr)


def fit_poisson_regression(train_file_name):
    data = pd.read_csv(train_file_name)

    # df = df[np.isfinite(df['EPS'])]

    ed_reg_time = pd.to_datetime(data["EDREGTIME"])
    ed_out_time = pd.to_datetime(data["EDOUTTIME"])

    # print(data["EDREGTIME"][0:5])

    # print(ed_reg_time[0:5])
    # print(ed_out_time[0:5])

    er_duration = (pd.to_datetime(data["EDOUTTIME"], infer_datetime_format=True) -
                   pd.to_datetime(data["EDREGTIME"], infer_datetime_format=True))

    er_duration = er_duration / np.timedelta64(1, 'h')

    # print(er_duration)

    data["er_duration"] = er_duration

    average_dosage_days = (data["sum_dosage_days"] * 24.0) / data["er_duration"]

    # print(data["sum_dosage_days"][0:5]*24.0)
    # print(data["er_duration"][0:5])

    # print(average_dosage_days[0:5])
    data["average_dosage_days"] = average_dosage_days

    data = data[np.isfinite(data['average_dosage_days'])]
    #
    # full_days = []
    # for i in average_dosage_days:
    #     full_days.append(round(i))
    #
    #     # try:
    #     #     i = math.ceil(i)
    #     # except Exception:
    #     #     i = 0

    full_average_days = [math.ceil(n) for n in list(data["average_dosage_days"])]
    # [n * 2 for n in numbers if n % 2 == 1]

    print("type = ", type(average_dosage_days))  # , type(full_average_days))
    data["full_average_dosage_days"] = full_average_days

    admit_duration = (pd.to_datetime(data["DISCHTIME"], infer_datetime_format=True) -
                      pd.to_datetime(data["ADMITTIME"], infer_datetime_format=True))

    admit_duration = admit_duration / np.timedelta64(1, 'h')
    data["admit_duration"] = admit_duration

    edited_eths = []
    for i in data["ETHNICITY"]:
        if i == "BLACK/AFRICAN AMERICAN":
            i = "black"
        if i == "HISPANIC OR LATINO":
            i = "hispanic_or_latino"
        if i == "HISPANIC/LATINO - CUBAN":
            i = "hispanic_latino_cuban"
        if i == "WHITE":
            i = "white"
        # i = i.lower().rstrip()
        #
        # if " " in i:
        #     i = i.replace(" ", "_")
        edited_eths.append(i)
    data["edited_eths"] = edited_eths

    adjusted_age = data['age'] / max(data['age'])
    data['adjusted_age'] = adjusted_age

    print(data.head())

    # expr = """average_dosage_days ~ gender  + INSURANCE + age + icd9_code + admit_duration + DISCHARGE_LOCATION"""
    # expr = """average_dosage_days ~ gender  + INSURANCE + age + icd9_code + admit_duration + DISCHARGE_LOCATION"""
    # expr = """full_average_dosage_days ~ edited_eths"""
    expr = """full_average_dosage_days ~ gender  + INSURANCE + adjusted_age + icd9_code + admit_duration + DISCHARGE_LOCATION +
    MARITAL_STATUS"""
    # expr = """average_dosage_days ~ gender  + INSURANCE + age + icd9_code + admit_duration + DISCHARGE_LOCATION +
    # MARITAL_STATUS + ETHNICITY"""
    # expr = """average_dosage_days ~ gender  + INSURANCE + age + icd9_code + admit_duration + RELIGION +
    # MARITAL_STATUS + ETHNICITY"""
    # expr = """average_dosage_days ~ gender  + INSURANCE + age + icd9_code + admit_duration + DISCHARGE_LOCATION"""

    # ethnicity_dummies = pd.get_dummies(data["edited_eths"], prefix="ethnicity").iloc[:, :-1]
    # print(ethnicity_dummies)
    #
    # train_df = data["full_average_dosage_days"]
    # train_df = pd.concat([train_df, ethnicity_dummies], axis=1)

    # print(train_df.head())

    # Set up the X and y matrices
    # y_train, X_train = dmatrices(expr, train_df, return_type='dataframe')
    y_train, X_train = dmatrices(expr, data, return_type='dataframe')

    print("X_train = ", X_train.head())
    poisson_training_results = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

    linear_regression_training_results = sm.OLS(y_train, X_train).fit()
    # print("linear regression results = ", linear_regression_training_results.summary())

    # Print the training summary.
    print(poisson_training_results.summary())

    # linear_reg_training_results = LinearRegression().fit(X_train, y_train)
    # print(linear_reg_training_results.score(X_train, y_train))
    # #
    # print("Linear Reg Score = ", linear_reg_training_results.score(X_train, y_train))
    # print("Linear Reg Coefs = ", linear_reg_training_results.coef_)
    # print("Linear Reg Intercept = ", linear_reg_training_results.intercept_)
    #       linear_reg_training_results.intercept_)

    # create regressor object
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # fit the regressor with x and y data
    regression_results = regressor.fit(X_train, y_train)
    print("Random Forest Feature Importances = ", regression_results.feature_importances_)
    print("Most important feature = ", max(regression_results.feature_importances_),
          list(regression_results.feature_importances_).index(max(regression_results.feature_importances_)))
    print("Random Forest Feature Importances sorted in reverse = ",
          sorted(list(regression_results.feature_importances_)))

    print(data.head(5))
    data.to_csv("./data/features.csv")


def plot_frequencies(data_filename, data_column_name):
    data = pd.read_csv(data_filename)
    frequencies = pd.value_counts(data[data_column_name].values, sort=True)
    fig, ax = plt.subplots()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    x = list(frequencies.keys())
    y = list(frequencies)

    rect = ax.bar(x, y)
    ax.set_title(data_column_name.capitalize() + " frequencies ")
    ax.set_ylabel("Count")
    ax.set_xlabel(data_column_name.capitalize())
    plt.xticks(rotation=90)
    # ax.set_xticks(x)
    # ax.xticklabels(x, rotation=45)
    plt.tight_layout()

    autolabel(rect)
    plt.savefig("./data/visualization/" + data_column_name.lower() + "_frequencies.png", dpi=300, bbox_inches='tight')
    plt.show()

    # data[['col1', 'col2', 'col3', 'col4']].groupby(['INSURANCE', 'gender']).agg(['count', 'count'])
    # h = data[['gender', 'INSURANCE', 'ETHNICITY', 'RELIGION']].groupby(['INSURANCE', 'gender']).agg(['count'])
    # h = data.groupby(['INSURANCE', 'gender']).agg(['count'])
    # h = data[['Unnamed: 0', 'INSURANCE', 'gender']].groupby(['INSURANCE', 'gender']).agg('count')
    # print(h)


def plot_multi_category_frequencies(data_filename, column_name_1, column_name_2):
    data = pd.read_csv(data_filename)
    column_1_frequencies = pd.value_counts(data[column_name_1].values, sort=True)

    labels = list(column_1_frequencies.keys())
    print(labels)

    h = data[['Unnamed: 0', column_name_1, column_name_2]].groupby([column_name_1, column_name_2]).agg('count')
    print(h, type(h))

    series = h["Unnamed: 0"]

    f_counts = []
    m_counts = []

    for label in labels:
        try:
            female_key_index = list(series.keys()).index((label, 'F'))
            f_count = list(series)[female_key_index]
        except Exception:
            f_count = 0

        try:
            male_key_index = list(series.keys()).index((label, 'M'))
            m_count = list(series)[male_key_index]
        except Exception:
            m_count = 0

        f_counts.append(f_count)
        m_counts.append(m_count)

    print("f_counts = ", f_counts)
    print("m_counts = ", m_counts)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, m_counts, width, label='Male')
    rects2 = ax.bar(x + width / 2, f_counts, width, label='Female')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequencies')
    ax.set_title('Frequencies by ' + column_name_1.lower() + ' and ' + column_name_2.lower())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    fig.savefig("./data/visualization/" + column_name_1.lower() + "_and_" + column_name_2.lower() + "_frequencies.png")
    plt.show()


def plot_outcome_against_features(data_file_name, independent_variable, outcome_variable):
    data = pd.read_csv(data_file_name)

    # plt.scatter_max_n=1(data[independent_variable].astype(str), data[outcome_variable])
    # plt.margins(x=0.5)

    sns.swarmplot(x=independent_variable, y=outcome_variable, data=data)

    plt.show()


def plot_histograms(data_file):
    data = pd.read_csv(data_file)
    dosage_gender_insurance = data[["full_average_dosage_days", "gender", "INSURANCE"]]
    full_dosage_days = data["full_average_dosage_days"]

    fig, ax = plt.subplots()
    ax.hist(np.log(full_dosage_days),
            color='blue', edgecolor='black')
    ax.set_title('Histogram of full average dosage days')
    ax.set_xlabel('Full average dosage days - (log)')
    ax.set_ylabel("Frequency")

    fig.savefig("./data/visualization/full_dosage_days_histogram.png")
    #
    # # Title and labels
    # ax.set_title('Histogram with Binwidth = %d' % 1, size=30)
    # ax.set_xlabel('Delay (min)', size=22)
    # ax.set_ylabel('Flights', size=22)

    # insurance_keys = list(dosage_gender_insurance["INSURANCE"].value_counts().keys())
    # print("insurance_keys = ", insurance_keys)
    #
    # keys_dosages = []
    # for key in insurance_keys:
    #     print("key = ", key)
    #
    #     key_dosages = list(dosage_gender_insurance["full_average_dosage_days"].\
    #         where(dosage_gender_insurance["INSURANCE"] == key).dropna())
    #
    #     print(key_dosages)
    #
    #     # key_dosages = list(dosage_gender_insurance[["INSURANCE"] == key]["full_average_dosage_days"])
    #     keys_dosages.append(key_dosages)
    #
    # # Assign colors for each airline and the names
    # colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
    # names = list(insurance_keys)
    #
    # plt.hist(keys_dosages, bins=15, normed=True,
    #          color=colors, label=names)
    #
    # # Plot formatting
    # plt.legend()
    # plt.xlabel('Delay (min)')
    # plt.ylabel('Normalized Flights')
    # plt.title('Side-by-Side Histogram with Multiple Airlines')
    plt.show()


def plot_box_plot(data_file):
    data = pd.read_csv(data_file)
    male_averages = data["average_dosage_days"].where(data["gender"] == "M").dropna()
    female_averages = data["average_dosage_days"].where(data["gender"] == "F").dropna()

    private_insurance_averages = data["average_dosage_days"].where(data["INSURANCE"] == "Private").dropna()
    medicaid_insurance_averages = data["average_dosage_days"].where(data["INSURANCE"] == "Medicaid").dropna()
    medicare_insurance_averages = data["average_dosage_days"].where(data["INSURANCE"] == "Medicare").dropna()
    government_insurance_averages = data["average_dosage_days"].where(data["INSURANCE"] == "Government").dropna()
    self_pay_insurance_averages = data["average_dosage_days"].where(data["INSURANCE"] == "Self Pay").dropna()

    print(male_averages)

    fig7, ax7 = plt.subplots()
    ax7.set_title('Average dosage days per gender')
    # ax7.boxplot([male_averages, female_averages])
    ax7.boxplot([np.log(male_averages), np.log(female_averages)])
    ax7.set_xlabel('gender')
    ax7.set_ylabel('Average dosage days (log)')
    # ax7.set_xticks([1, 2], ['M', 'F'])
    plt.xticks([1, 2], ['M', 'F'])
    fig7.savefig("./data/visualization/average_dosage_days_per_gender.png")

    fig8, ax8 = plt.subplots()
    ax8.set_title('Average dosage days per insurance')
    # ax8.boxplot([private_insurance_averages, medicaid_insurance_averages, medicare_insurance_averages,
    #              government_insurance_averages, self_pay_insurance_averages])
    ax8.boxplot([np.log(private_insurance_averages), np.log(medicaid_insurance_averages),
                 np.log(medicare_insurance_averages),
                 np.log(government_insurance_averages), np.log(self_pay_insurance_averages)])
    ax8.set_xlabel('insurance')
    ax8.set_ylabel('Average dosage days (log)')
    plt.xticks([1, 2, 3, 4, 5], ['Private', 'Medicaid', 'Medicare', 'Government', 'Self Pay'])
    fig8.savefig("./data/visualization/average_dosage_days_per_insurance.png")

    plt.show()


def sum_days_by_date(filename):
    data = pd.read_csv(filename)
    d = data.groupby("HADM_ID")
    groups = d.groups
    g_keys = list(groups.keys())

    df = pd.DataFrame()
    for group in g_keys:
        f = d.get_group(group)
        grouped_by_date = f.groupby("startdate")
        date_groups = grouped_by_date.groups
        date_groups_keys = list(date_groups.keys())
        print("date group[ keys = ", date_groups_keys)

        print("group = ", group)
        for day in sorted(date_groups_keys):
            print("day = ", day)
            g = f.where(f["startdate"] == day)
            g = g[pd.notnull(g['HADM_ID'])]
            # h = g[g.columns[~g.isnull().all()]]
            # print("g = ", g, type(g))
            g_prime = g.head(1)
            narcotics = g.where(g["is_narcotic"] == "yes")
            narcotics = narcotics[pd.notnull(narcotics['HADM_ID'])]
            non_narcotics = g.where(g["is_narcotic"] == "no")
            non_narcotics = non_narcotics[pd.notnull(non_narcotics['HADM_ID'])]
            narcotic_days = sum(narcotics["dosage_days"])
            non_narcotics_days = sum(non_narcotics["dosage_days"])
            print("sum dosage days = ", sum(g["dosage_days"]))
            print("sum narcotic_dosage days = ", narcotic_days)
            print("sum non_narcotic_dosage days = ", non_narcotics_days)

            g_prime["sum_dosage_days"] = sum(g["dosage_days"])
            g_prime["sum_narcotic_dosage_days"] = narcotic_days
            g_prime["sum_non_narcotic_dosage_days"] = non_narcotics_days

            g_prime.drop("dosage_days", axis=1)
            # f_prime.insert(7, "sum_dosage_days", days)
            # f_prime.insert(8, "sum_narcotics_dosage_days", narcotic_days)
            # f_prime.insert(9, "sum_non_narcotics_dosage_days", non_narcotics_days)
            gender = list(g_prime["gender"])[0]
            if gender == "M":
                gender_num = 0
            else:
                gender_num = 1
            g_prime["gender_num"] = gender_num
            # f_p = f_prime[7:]
            # print("f_p = ", f_p)
            start_date_index = g_prime.columns.get_loc("startdate")
            df = pd.concat([df, g_prime.iloc[:, :]])
        print(df.head())
    df.to_csv("./data/prescriptions_with_summed_days_grouped_by_date.csv")


def plot_stacked_bar_chart(data_file):
    # read_first_prescriptions(data_file)
    # print(data.head())

    # assign_dosage_days(data_file)
    # print(data.head())

    # sum_days_by_date("./data/prescriptions_with_dosage_days_computed.csv")

    data = pd.read_csv("./data/prescriptions_with_summed_days_grouped_by_date.csv")
    data = data.sort_values(by='sum_dosage_days', ascending=False)
    hadm_ids = data["HADM_ID"].unique()

    print(hadm_ids[0:3])

    for hadm_id in list(hadm_ids):
        print(hadm_id)
        group_1 = data.where(data["HADM_ID"] == hadm_id)
        group_1 = group_1[pd.notnull(group_1['HADM_ID'])]
        narcs = []
        non_narcs = []
        print(list(group_1["startdate"]))
        disch_time = pd.to_datetime(group_1['DISCHTIME'], format="%Y-%m-%d %H:%M:%S")
        admit_time = pd.to_datetime(group_1['ADMITTIME'], format="%Y-%m-%d %H:%M:%S")

        # admit_duration = pd.Timedelta(disch_time - admit_time).seconds / 3600.0
        admit_duration = math.floor(list((disch_time - admit_time).astype('timedelta64[h]'))[0] / 24.0)

        gender = list(group_1["gender"])[0]
        age = list(group_1["age"])[0]

        # print(admit_duration, gender, age)

        dates = sorted(list(group_1["startdate"]))

        for start_date in dates:
            # print("startdate = ", start_date)
            d = group_1.where(group_1["startdate"] == start_date)
            d = d[pd.notnull(d['HADM_ID'])]
            narc_values = sum(d["sum_narcotic_dosage_days"])
            non_narc_values = sum(d["sum_non_narcotic_dosage_days"])
            narcs.append(narc_values)
            non_narcs.append(non_narc_values)

        # print("narcs = ", narcs)
        # print("non_narcs = ", non_narcs)

        N = len(dates)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.45  # the width of the bars: can also be len(x) sequence
        # width = len(dates)  # the width of the bars: can also be len(x) sequence

        p1 = plt.bar(ind, narcs, width)  # , yerr=menStd)
        p2 = plt.bar(ind, non_narcs, width, bottom=narcs)  # , yerr=womenStd)

        plt.ylabel('Dosage days')
        plt.title('Dosage days by date for each admission')
        plt.xticks(ind, dates)
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.yticks(np.arange(0, 100, 10))
        plt.legend((p1[0], p2[0]), ('Narcs', 'Non_narcs'))
        # plt.title("Dosage Values for patient with admission no:  "+str(hadm_id) + ".\n Pt Age = "+str(age)+
        #           " G = "+gender + " admitted for "+str(admit_duration)+ " full days")
        plt.title("hadm_id = " + str(hadm_id) + ", age = " + str(age) +
                  ", g = " + gender + ", admit duration = " + str(admit_duration) + " full days")
        plt.savefig("./data/results/dosages_over_time/" + str(list(hadm_ids).index(hadm_id) + 1) + "_adm_id_" + str(
            hadm_id) + ".png")
        plt.show()


def plot_scatter_dosage_gender_against_duration(start_index, end_index):
    data = pd.read_csv("./data/prescriptions_with_summed_days_grouped_by_date.csv")
    d = data.groupby("HADM_ID")
    groups = d.groups
    g_keys = list(groups.keys())

    df = pd.DataFrame()
    for group in g_keys:
        f = d.get_group(group)
        dates = sorted(list(f["startdate"]))
        g = pd.DataFrame()
        # get dates for each group and sort them asc. get the each date and get their sum_dosage
        # append each date's df to a larger df; where len(dates)>1
        for start_date in dates[start_index:end_index]:
            h = f.where(f["startdate"] == start_date)
            h = h[pd.notnull(h['HADM_ID'])]
            g = pd.concat([g, h])
        g_prime = g.head(1)
        # print("g_prime = ", g_prime)
        disch_time = pd.to_datetime(g_prime['DISCHTIME'], format="%Y-%m-%d %H:%M:%S")
        admit_time = pd.to_datetime(g_prime['ADMITTIME'], format="%Y-%m-%d %H:%M:%S")
        # admit_duration = math.floor(list((disch_time - admit_time).astype('timedelta64[h]'))[0] / 24.0)
        admit_duration = list((disch_time - admit_time).astype('timedelta64[h]'))[0] / 24.0
        # print("admit_duration ", [admit_duration])
        g_prime["admit_duration"] = [admit_duration]
        g_prime["sum_sum_narcotic_dosage_days"] = sum(g["sum_narcotic_dosage_days"])
        g_prime["sum_sum_non_narcotic_dosage_days"] = sum(g["sum_non_narcotic_dosage_days"])
        g_prime["sum_sum_dosage_days"] = sum(g["sum_dosage_days"])
        df = pd.concat([df, g_prime[
            ["HADM_ID", "sum_sum_narcotic_dosage_days", "sum_sum_non_narcotic_dosage_days", "gender", "admit_duration",
             "sum_sum_dosage_days"]]])

    # print(len(df))
    # print(df)
    df_women = df.where(df["gender"] == "F")
    df_women = df_women[pd.notnull(df_women['gender'])]
    df_men = df.where(df["gender"] == "M")
    df_men = df_men[pd.notnull(df_men['gender'])]

    # female_narcs_averages, female_non_narcs_averages, female_total_averages = \
    #     compute_dosage_days_averages_per_admit_period(df_women, "admit_duration",
    #                                                   [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30],
    #                                                    [31, 35],
    #                                                    [36, 40], [41, 45], [46, 50], [51, 55]])
    female_narcs_averages, female_non_narcs_averages, female_total_averages = \
        compute_dosage_days_averages_per_admit_period(df_women, "admit_duration",
                                                      [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30],
                                                       [31, 35],
                                                       [36, 40], [41, 45], [46, 50], [51, 55], [56, 60], [61, 65],
                                                       [66, 70], [71, 75], [76, 80], [81, 85]])
    # male_narcs_averages, male_non_narcs_averages, male_total_averages = \
    #     compute_dosage_days_averages_per_admit_period(df_men, "admit_duration",
    #                                                   [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30],
    #                                                    [31, 35],
    #                                                    [36, 40], [41, 45], [46, 50], [51, 55]])
    male_narcs_averages, male_non_narcs_averages, male_total_averages = \
        compute_dosage_days_averages_per_admit_period(df_men, "admit_duration",
                                                      [[0, 5], [6, 10], [11, 15], [16, 20], [21, 25], [26, 30],
                                                       [31, 35],
                                                       [36, 40], [41, 45], [46, 50], [51, 55], [56, 60], [61, 65],
                                                       [66, 70], [71, 75], [76, 80], [81, 85]])

    # plot narcotic and non_narcotic averages for each gender, label gender and drug type
    fig1, axes1 = plt.subplots()
    # axes1.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55], female_narcs_averages, marker='^', c="g",
    #               label="female_narcs")
    axes1.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], female_narcs_averages,
                  marker='^', c="g",
                  label="female_narcs")
    axes1.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], male_narcs_averages, marker='^',
                  c="r",
                  label="male_narcs")
    axes1.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], female_non_narcs_averages,
                  marker='o', c="g",
                  label="female_narcs")
    axes1.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], male_non_narcs_averages,
                  marker='o', c="r",
                  label="male_narcs")
    axes1.set_ylabel("Average Dosage Days Per Group's Admit Duration")
    axes1.set_xlabel("Admit Duration (Days)")
    axes1.set_title("Admit Duration Group's Average Pain Relief Dosage Against Admit_Duration +\n For Both Genders: n="
                    + str(end_index - start_index))
    axes1.legend()
    # fig1.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n/n=" + str(
    #     end_index - start_index) +
    #              "_scatter_plot_average_narc_and_non_narc_dosages_per_admission_group_against_duration.png")
    fig1.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/n=" + str(
        end_index - start_index) +
                 "_scatter_plot_average_narc_and_non_narc_dosages_per_admission_group_against_duration.png")
    # plt.show()
    # plt.close()

    # plot average dosages for each admit_duration group, label by gender
    fig2, axes2 = plt.subplots()
    axes2.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], female_total_averages,
                  marker='^', c="g",
                  label="female")
    axes2.scatter([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85], male_total_averages, marker='^',
                  c="r",
                  label="male")
    axes2.set_ylabel("Average Dosage Days Per Group's Admit Duration")
    axes2.set_xlabel("Admit Duration (Days)")
    axes2.set_title("Admit Duration Group's Average Pain Relief Dosage Against Admit_Duration +\n For Both Genders: n="
                    + str(end_index - start_index))
    axes2.legend()
    # fig2.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n/sum_of_sum_dosage_days/n=" + str(
    #     end_index - start_index) + "_scatter_plot_average_dosages_per_admission_group_against_duration.png")
    fig2.savefig(
        "./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/sum_of_sum_dosage_days/n=" + str(
            end_index - start_index) + "_scatter_plot_average_dosages_per_admission_group_against_duration.png")
    # plt.show()
    # plt.close()

    # plot narcotic and non_narcotic dosages for each gender, label gender and drug type
    fig3, axes3 = plt.subplots()
    axes3.scatter(df_women["admit_duration"], df_women["sum_sum_narcotic_dosage_days"], marker='^', c="g",
                  label="female_narc")
    axes3.scatter(df_men["admit_duration"], df_men["sum_sum_narcotic_dosage_days"], marker='^', c="r",
                  label="male_narc")
    axes3.scatter(df_women["admit_duration"], df_women["sum_sum_non_narcotic_dosage_days"], marker='o', c="g",
                  label="female_non-narc")
    axes3.scatter(df_men["admit_duration"], df_men["sum_sum_non_narcotic_dosage_days"], marker='o', c="r",
                  label="male_non-narc")
    axes3.set_ylabel("Dosage Days")
    axes3.set_xlabel("Admit Duration (Days)")
    axes3.set_title("Pain Relief Dosage Against Admit_Duration For Both Genders: n=" + str(end_index - start_index))
    axes3.legend()
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 8])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=8/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=8/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 6])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=6/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=6/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 5])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=5/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=5/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 4])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=4/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=4/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 3])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=3/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=3/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 3])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=2/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=2/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    axes3.set_ylim([0, 1])
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n=1/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n=1/n=" + str(
        end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    # plt.show()
    # plt.close()

    # plot narc and non-narcs values for each gender separately
    # female
    fig4, axes4 = plt.subplots()
    axes4.scatter(df_women["admit_duration"], df_women["sum_sum_narcotic_dosage_days"], marker='^', c="g", label="narc")
    axes4.scatter(df_women["admit_duration"], df_women["sum_sum_non_narcotic_dosage_days"], marker='o', c="r",
                  label="non-narc")
    axes4.set_ylabel("Dosage Days")
    axes4.set_xlabel("Admit Duration (Days)")
    axes4.set_title("Pain Relief Dosage Against Admit_Duration For Females: n=" + str(end_index - start_index))
    axes4.legend()
    # fig4.savefig(
    #     "./data/results/dosages_over_time/scatter_plots/scatter_max_n/female_only_narcs_and_non_narcs/n=" + str(
    #         end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig4.savefig(
        "./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/female_only_narcs_and_non_narcs/n=" + str(
            end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    # plt.show()
    # plt.close()

    # male
    fig5, axes5 = plt.subplots()
    axes5.scatter(df_men["admit_duration"], df_men["sum_sum_narcotic_dosage_days"], marker='^', c="g", label="narc")
    axes5.scatter(df_men["admit_duration"], df_men["sum_sum_non_narcotic_dosage_days"], marker='o', c="r",
                  label="non-narc")
    axes5.set_ylabel("Dosage Days")
    axes5.set_xlabel("Admit Duration (Days)")
    axes5.set_title("Pain Relief Dosage Against Admit_Duration For Males: n=" + str(end_index - start_index))
    axes5.legend()
    # fig5.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n/male_only_narcs_and_non_narcs/n=" + str(
    #     end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    fig5.savefig(
        "./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/male_only_narcs_and_non_narcs/n=" + str(
            end_index - start_index) + "_scatter_plot_dosage_against_duration.png")
    # plt.show()
    # plt.close()

    # plot sum_dosage_days against admission duration
    fig6, axes6 = plt.subplots()
    axes6.scatter(df_women["admit_duration"], df_women["sum_sum_dosage_days"], c="g", label="female")
    axes6.scatter(df_men["admit_duration"], df_men["sum_sum_dosage_days"], c="r", label="male")
    axes6.set_ylabel("Dosage Days")
    axes6.set_xlabel("Admit Duration (Days)")
    axes6.set_title(
        "Total Pain Relief Dosage Against Admit_Duration For Both Genders: n=" + str(end_index - start_index))
    axes6.legend()
    # fig6.savefig("./data/results/dosages_over_time/scatter_plots/scatter_max_n/n=" + str(
    #     end_index - start_index) + "_sum_of_sums_scatter_plot_dosage_against_duration.png")
    fig6.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/scatter_max_n/n=" + str(
        end_index - start_index) + "_sum_of_sums_scatter_plot_dosage_against_duration.png")
    plt.show()
    # plt.close()

    # plt.show()


def compute_dosage_days_averages_per_admit_period(dataframe, entity, range_list):
    average_narcs = []
    average_non_narcs = []
    average_totals = []
    for range in range_list:
        # print("range = ", range)
        range_start = range[0]
        range_end = range[1]
        df = dataframe.where(dataframe[entity].between(range_start, range_end, inclusive=True))
        df = df[pd.notnull(df[entity])]

        # print("len of df = ", len(df))
        if len(df) > 0:
            average_narc_dosage_days = sum(df["sum_sum_narcotic_dosage_days"]) / len(df)
            average_non_narc_dosage_days = sum(df["sum_sum_non_narcotic_dosage_days"]) / len(df)
            average_total_dosage_days = sum(df["sum_sum_dosage_days"]) / len(df)
        else:
            average_narc_dosage_days = 0
            average_non_narc_dosage_days = 0
            average_total_dosage_days = 0
        # print(df)

        # print(average_narc_dosage_days, average_non_narc_dosage_days, average_total_dosage_days)
        average_narcs.append(average_narc_dosage_days)
        average_non_narcs.append(average_non_narc_dosage_days)
        average_totals.append(average_total_dosage_days)
    return average_narcs, average_non_narcs, average_totals


# def mean(a):
#     return sum(a) / len(a)
# a = [[240, 240, 239],
#      [250, 249, 237],
#      [242, 239, 237],
#      [240, 234, 233]]

def mean(a):
    count = len([i for i in a if i > 0])
    print("count = ", count)
    #     count = 0
    #     for i in a:
    #         for j in i:
    #             if j > 0:
    #                 count = count + 1
    #
    # return sum(a) / len(a)
    try:
        return sum(a) / count
    except ZeroDivisionError:
        return 0


def std(a):
    average = list(map(mean, zip(*a)))
    print("a = ", a)
    a_zip = zip(*a)
    a_zip = list(a_zip)
    print(a_zip)
    print(len(a_zip))
    devs = []
    for i in range(0, len(a_zip)):
        vals = a_zip[i]
        print("vals = ", vals)
        vals = [j for j in vals if j > 0]
        average_val = average[i]
        diff = np.subtract(vals, average_val)
        diff_square = [i ** 2 for i in diff]
        diff_square_sum = sum(diff_square)
        dev = math.sqrt(diff_square_sum / float(len(vals)))
        devs.append(dev)
        print("dev = ", dev)
    return devs


def dosage_days_per_admission_period(filename):
    data = pd.read_csv(filename)
    hadm_ids = list(data["HADM_ID"].unique())

    all_dosages = []
    genders = []
    admit_durations = []
    for hadm_id in hadm_ids:
        dosages = []
        print("hadm_id = ", hadm_id)
        hadm_id_records = data.where(data["HADM_ID"] == hadm_id)
        hadm_id_records = hadm_id_records[pd.notnull(hadm_id_records["HADM_ID"])]

        gender = list(hadm_id_records["gender"])[0]
        print("gender = ", gender)
        genders.append(gender)

        # admit_duration = list(hadm_id_records["admit_duration"])[0]

        # print(len(hadm_id_records))

        admittime = list(hadm_id_records["ADMITTIME"])[0]
        dischargetime = list(hadm_id_records["DISCHTIME"])[0]

        # print(type(admittime), type(dischargetime))
        # print(admittime)

        admittime = pd.to_datetime(admittime, format="%Y-%m-%d %H:%M:%S")
        dischargetime = pd.to_datetime(dischargetime, format="%Y-%m-%d %H:%M:%S")
        # print(admittime, type(admittime))
        # print(dischargetime, type(admittime))

        # for date in range(admittime, dischargetime):

        delta = dischargetime - admittime  # as timedelta
        admit_duration = delta.days
        print("admit_duration = ", admit_duration)
        admit_durations.append(admit_duration)

        for i in range(delta.days + 1):
            day = admittime + timedelta(days=i)
            # print(day)
            # print("date = ", day)

            # to_datetime
            day_records = hadm_id_records.where(
                pd.to_datetime(hadm_id_records["startdate"]).dt.date == datetime.datetime.date(day))
            day_records = day_records[pd.notnull(day_records["HADM_ID"])]
            # print("day records = ", day_records)
            if len(day_records) == 0:
                dosage_days = 0
            else:
                dosage_days = list(day_records["sum_dosage_days"])[0]
            # print("dosage_days = ", dosage_days)
            dosages.append(dosage_days)
        all_dosages.append(dosages)
    # print("all dosages = ", all_dosages)

    # g = all_dosages[0]
    # print("g before = ", g, len(g))
    # g = (g + [0] * 3)[:3]
    # print("g after = ", g, len(g))

    female_dosages = []
    male_dosages = []
    # print(" max dosage in all dosages = ", max(all_dosages, key=len))
    # print(" max-dosage-in-all-dosages's len = ", len(max(all_dosages, key=len)))
    padded_all_dosages = []
    fig1, axes1 = plt.subplots()

    for i in range(0, len(all_dosages)):
        gender = genders[i]
        dosages = all_dosages[i]
        dosages = (dosages + [0] * 86)[:86]
        padded_all_dosages.append(dosages)
        if gender == 'F':
            axes1.scatter(list(range(1, 1 + 30)), dosages[:30], marker='^', c="g", label="female")
            female_dosages.append(dosages)
        if gender == 'M':
            axes1.scatter(list(range(1, 1 + 30)), dosages[:30], marker='^', c="r", label="male")
            male_dosages.append(dosages)
    print("all dosages = ", padded_all_dosages[0:3])
    # day_averages = mean(padded_all_dosages)
    day_averages = map(mean, zip(*padded_all_dosages))
    day_averages = list(day_averages)
    print("day averages =", day_averages)

    print("len of x = ", len(list(range(1, 1 + 30))))

    print("len of y = ", len(day_averages[:30]))

    axes1.scatter(list(range(1, 1 + 30)), day_averages[:30], marker='^', c="b", label="average")
    axes1.set_ylim([0.0001, 15])
    # axes1.legend()
    # axes1.set_ylim([0.001, 5])

    plt.show()


def dosage_days_per_admission_period_2(filename):
    data = pd.read_csv(filename)
    hadm_ids = list(data["HADM_ID"].unique())

    all_dosages = []
    genders = []
    are_narcotics = []
    admit_durations = []
    for hadm_id in hadm_ids:
        dosages = []
        dosages1 = []
        print("hadm_id = ", hadm_id)
        hadm_id_records = data.where(data["HADM_ID"] == hadm_id)
        hadm_id_records = hadm_id_records[pd.notnull(hadm_id_records["HADM_ID"])]

        gender = list(hadm_id_records["gender"])[0]
        print("gender = ", gender)
        genders.append(gender)

        is_narcotic = list(hadm_id_records["is_narcotic"])[0]
        are_narcotics.append(is_narcotic)

        # print(len(hadm_id_records))

        admittime = list(hadm_id_records["ADMITTIME"])[0]
        dischargetime = list(hadm_id_records["DISCHTIME"])[0]

        # print(type(admittime), type(dischargetime))
        # print(admittime)

        admittime = pd.to_datetime(admittime, format="%Y-%m-%d %H:%M:%S")
        dischargetime = pd.to_datetime(dischargetime, format="%Y-%m-%d %H:%M:%S")

        delta = dischargetime - admittime  # as timedelta
        admit_duration = delta.days + 1
        print("admit_duration = ", admit_duration)
        admit_durations.append(admit_duration)

        for i in range(delta.days + 1):
            day = admittime + timedelta(days=i)
            print("date = ", day)

            day_records = hadm_id_records.where(
                pd.to_datetime(hadm_id_records["startdate"]).dt.date == datetime.datetime.date(day))
            day_records = day_records[pd.notnull(day_records["HADM_ID"])]
            day_records = day_records[pd.notnull(day_records["enddate"])]
            # print(list(day_records["enddate"]))
            # print("testing = ", pd.to_datetime(day_records["enddate"]).dt.date - datetime.datetime.date(day))
            if len(day_records) == 0:
                dosage_days = 0
                dosages.append({"date": day, "dosage_days": dosage_days})
            else:
                start_end_delta = (pd.to_datetime(day_records["enddate"]).dt.date - datetime.datetime.date(day)).dt.days
                # print("start end delta = ", list(start_end_delta), type(start_end_delta))
                for j in range(list(start_end_delta)[0] + 1):
                    pday = day + timedelta(days=j)
                    t_dosage_days = list(day_records["sum_dosage_days"])[0]
                    dosage_days = (list(day_records["sum_dosage_days"])[0]) / (list(start_end_delta)[0] + 1)
                    # print("pday = ", pday, t_dosage_days, dosage_days)
                    dosages.append({"date": pday, "dosage_days": dosage_days})
            # print("dosages = ", dosages)
            from collections import defaultdict
            c = defaultdict(int)
            for d in dosages:
                c[d['date']] += d['dosage_days']
            # print("summed dosages = ", c, list(c.values()))
            dosages1 = list(c.values())
            # dosages.append(dosage_days)
            # print("dosage_days = ", dosage_days)
        all_dosages.append(dosages1)
    print("all dosages = ", all_dosages)
    return all_dosages, genders, are_narcotics, admit_durations


def graph(all_dosages, genders, are_narcotics, admit_durations, max_range):
    female_dosages = []
    male_dosages = []
    narc_dosages = []
    non_narc_dosages = []
    female_narc_dosages = []
    female_non_narc_dosages = []
    male_narc_dosages = []
    male_non_narc_dosages = []
    print(" max dosage in all dosages = ", max(all_dosages, key=len))
    print(" max-dosage-in-all-dosages's len = ", len(max(all_dosages, key=len)))

    max_admit_length = len(max(all_dosages, key=len))

    padded_all_dosages = []
    fig1, axes1 = plt.subplots()
    fig2, axes2 = plt.subplots()
    fig3, axes3 = plt.subplots()
    fig4, axes4 = plt.subplots()
    fig5, axes5 = plt.subplots()

    for i in range(0, len(all_dosages)):
        gender = genders[i]
        is_narcotic = are_narcotics[i]
        dosages = all_dosages[i]
        dosages = (dosages + [0] * max_admit_length)[:max_admit_length]
        padded_all_dosages.append(dosages)
        if gender == 'F':
            axes1.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="g", label="female")
            female_dosages.append(dosages)
        if gender == 'M':
            axes1.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="r", label="male")
            male_dosages.append(dosages)
        if is_narcotic == "yes":
            axes2.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="r", label="narcs")
            narc_dosages.append(dosages)
        if is_narcotic == "no":
            axes2.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="g", label="non-narcs")
            non_narc_dosages.append(dosages)

        if gender == 'F' and is_narcotic == "yes":
            axes3.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="g", label="female-narcs")
            female_narc_dosages.append(dosages)
        if gender == 'F' and is_narcotic == "no":
            axes3.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='o', c="g",
                          label="female-non-narcs")
            female_non_narc_dosages.append(dosages)
        if gender == 'M' and is_narcotic == "yes":
            axes3.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='^', c="r", label="male-narcs")
            male_narc_dosages.append(dosages)
        if gender == 'M' and is_narcotic == "no":
            axes3.scatter(list(range(1, 1 + max_range)), dosages[:max_range], marker='o', c="r", label="male-non-narcs")
            male_non_narc_dosages.append(dosages)

    # print("all dosages = ", padded_all_dosages[0:3])
    # day_averages = mean(padded_all_dosages)
    day_averages = map(mean, zip(*padded_all_dosages))
    day_averages = list(day_averages)
    print("day averages =", day_averages)

    print("len of x = ", len(list(range(1, 1 + 30))))

    print("len of y = ", len(day_averages[:30]))

    narc_day_averages = map(mean, zip(*narc_dosages))
    narc_day_averages = list(narc_day_averages)

    non_narc_day_averages = map(mean, zip(*non_narc_dosages))
    non_narc_day_averages = list(non_narc_day_averages)

    day_dosage_stds = std(padded_all_dosages)

    axes1.scatter(list(range(1, 1 + max_range)), day_averages[:max_range], marker='^', c="b", label="average")
    axes1.set_ylim([0.0001, 12])
    axes1.scatter(list(range(1, 1 + max_range)), day_dosage_stds[:max_range], marker='^', c="orange",
                  label="stds")
    axes1.set_title("Daily Dosage Days by Gender")
    axes2.scatter(list(range(1, 1 + max_range)), day_averages[:max_range], marker='^', c="b", label="average")
    axes2.set_ylim([0.0001, 14])
    axes2.set_title("Daily Dosage Days by Medication Type")
    axes3.scatter(list(range(1, 1 + max_range)), narc_day_averages[:max_range], marker='^', c="lime",
                  label="narc-average")
    axes3.scatter(list(range(1, 1 + max_range)), non_narc_day_averages[:max_range], marker='o', c="cyan",
                  label="non-narc-average")
    axes3.set_ylim([0.0001, 13])
    axes3.set_title("Daily Dosage Days by Gender and Medication Type")

    female_day_averages = map(mean, zip(*female_dosages))
    female_day_averages = list(female_day_averages)

    male_day_averages = map(mean, zip(*male_dosages))
    male_day_averages = list(male_day_averages)

    axes4.scatter(list(range(1, 1 + max_range)), female_day_averages[:max_range], marker='^', c="g", label="female")
    axes4.scatter(list(range(1, 1 + max_range)), male_day_averages[:max_range], marker='^', c="r", label="male")
    axes4.scatter(list(range(1, 1 + max_range)), day_averages[:max_range], marker='^', c="b", label="all")
    axes4.set_title("Average Daily Dosage Days by Gender")

    female_narc_day_averages = map(mean, zip(*female_narc_dosages))
    female_narc_day_averages = list(female_narc_day_averages)
    female_non_narc_day_averages = map(mean, zip(*female_non_narc_dosages))
    female_non_narc_day_averages = list(female_non_narc_day_averages)

    male_narc_day_averages = map(mean, zip(*male_narc_dosages))
    male_narc_day_averages = list(male_narc_day_averages)
    male_non_narc_day_averages = map(mean, zip(*male_non_narc_dosages))
    male_non_narc_day_averages = list(male_non_narc_day_averages)

    axes5.scatter(list(range(1, 1 + max_range)), female_narc_day_averages[:max_range], marker='^', c="g",
                  label="female-narcs")
    axes5.scatter(list(range(1, 1 + max_range)), female_non_narc_day_averages[:max_range], marker='o', c="g",
                  label="female-non-narcs")
    axes5.scatter(list(range(1, 1 + max_range)), male_narc_day_averages[:max_range], marker='^', c="r",
                  label="male-narcs")
    axes5.scatter(list(range(1, 1 + max_range)), male_non_narc_day_averages[:max_range], marker='o', c="r",
                  label="male-non-narcs")
    axes5.scatter(list(range(1, 1 + max_range)), narc_day_averages[:max_range], marker='^', c="lime",
                  label="narc-average")
    axes5.scatter(list(range(1, 1 + max_range)), non_narc_day_averages[:max_range], marker='o', c="cyan",
                  label="non-narc-average")
    axes5.set_title("Average Daily Dosage Days by Gender and Medication Type")

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='^', color='w', label='female',
                              markerfacecolor='g', markersize=10),
                       Line2D([0], [0], marker='^', color='w', label='male',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], marker='^', color='w', label='average',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='^', color='w', label='std',
                              markerfacecolor='orange', markersize=10)]
    legend_elements_1 = [Line2D([0], [0], marker='^', color='w', label='non-narcs',
                                markerfacecolor='g', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='narcs',
                                markerfacecolor='r', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='average',
                                markerfacecolor='b', markersize=10)]

    legend_elements_2 = [Line2D([0], [0], marker='^', color='w', label='female',
                                markerfacecolor='g', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='male',
                                markerfacecolor='r', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='all',
                                markerfacecolor='b', markersize=10)]
    legend_elements_3 = [Line2D([0], [0], marker='^', color='w', label='female',
                                markerfacecolor='g', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='male',
                                markerfacecolor='r', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='narc-average',
                                markerfacecolor='lime', markersize=10),
                         Line2D([0], [0], marker='^', color='w', label='non-narc-average',
                                markerfacecolor='cyan', markersize=10)]
    legend_elements_4 = [Line2D([0], [0], marker='^', color='w', label='female-narc',
                                markerfacecolor='g', markersize=8.5),
                         Line2D([0], [0], marker='o', color='w', label='female-non-narc',
                                markerfacecolor='g', markersize=8.5),
                         Line2D([0], [0], marker='^', color='w', label='male-narc',
                                markerfacecolor='r', markersize=8.5),
                         Line2D([0], [0], marker='o', color='w', label='male-non-narc',
                                markerfacecolor='r', markersize=8.5),
                         Line2D([0], [0], marker='^', color='w', label='narc-average',
                                markerfacecolor='lime', markersize=8.5),
                         Line2D([0], [0], marker='o', color='w', label='non-narc-average',
                                markerfacecolor='cyan', markersize=8.5)]

    axes1.legend(handles=legend_elements)
    axes2.legend(handles=legend_elements_1)
    axes3.legend(handles=legend_elements_4)
    axes4.legend(handles=legend_elements_2)
    axes5.legend(handles=legend_elements_4)

    fig1.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/daily_dosage_days_by_gender.png")
    fig2.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/daily_dosage_days_by_medication_type.png")
    fig3.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/daily_dosage_days_by_gender_and_medication_type.png")
    fig4.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/average_daily_dosage_days_by_gender.png")
    fig5.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/average_daily_dosage_days_by_gender_and_medication_type.png")

    # fig1.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/daily_dosage_days_by_gender.png")
    # fig2.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/daily_dosage_days_by_medication_type.png")
    # fig3.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/daily_dosage_days_by_gender_and_medication_type.png")
    # fig4.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/average_daily_dosage_days_by_gender.png")
    # fig5.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/average_daily_dosage_days_by_gender_and_medication_type.png")

    # plt.show()

    df = pd.DataFrame(padded_all_dosages)
    count = df.astype(bool).sum(axis=0)
    print("non-zero values column-wise = ", list(count))

    df_f = pd.DataFrame(female_dosages)
    count_f = df_f.astype(bool).sum(axis=0)
    non_zero_f_count = len([i for i in count_f if i > 0])
    print("female non-zero values column-wise = ", list(count_f))

    df_m = pd.DataFrame(male_dosages)
    count_m = df_m.astype(bool).sum(axis=0)
    non_zero_m_count = len([i for i in count_m if i > 0])
    print("male non-zero values column-wise = ", list(count_m))

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    # Set position of bar on X axis
    # r1 = np.arange(len(count[:max_range]))
    r1 = np.arange(len(count_m[:max_range]))
    r2 = [x + barWidth for x in r1]
    # r3 = [x + barWidth for x in r2]

    # Make the plot
    # axes6.bar(r1, count[:max_range], color='b', width=barWidth, edgecolor='white', label='all')
    rects1 = axes6.bar(r1, round((count_m[:max_range] / len(male_dosages)) * 100.0, 1), color='r', width=barWidth,
                       edgecolor='white', label='male')
    rects2 = axes6.bar(r2, round((count_f[:max_range] / len(female_dosages)) * 100.0, 1), color='g', width=barWidth,
                       edgecolor='white', label='female')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axes6.annotate('{}'.format(height),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    # fig6.tight_layout()

    # Add xticks on the middle of the group bars
    axes6.set_xlabel('Admit day', fontweight='bold')
    print("ranges = ", range(1, max_range + 1))
    # axes6.set_xticks([r + barWidth for r in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    axes6.set_xticks([r + barWidth for r in range(max_range)])
    axes6.set_xticklabels(list(range(1, 1 + max_range)))

    # Create legend & Show graphic
    axes6.legend()
    axes6.set_ylabel("Percentage of patients", fontweight="bold")
    axes6.set_title("Percentage Patients Prescribed Pain Relief each Admit Day")
    fig6.savefig("./data/results/dosages_over_time/scatter_plots/heart_attack/all_admit_days_dosages"
                 "/bar_chart_of_gender_numbers_per_admit_day.png")
    # fig6.savefig("./data/results/dosages_over_time/scatter_plots/pneumonia/all_admit_days_dosages"
    #              "/bar_chart_of_gender_numbers_per_admit_day.png")

    fig7, axes7 = plt.subplots()
    # day_dosage_stds = std(padded_all_dosages)
    # axes7.scatter(list(range(1, 1 + max_range)), day_dosage_stds[:max_range], marker='x', c="orange",
    #               label="stds")

    f_count = len([i for i in female_day_averages[:max_range] if i > 0])
    m_count = len([i for i in male_day_averages[:max_range] if i > 0])
    a_count = len([i for i in day_averages[:max_range] if i > 0])

    fem_av = sum(female_day_averages[:max_range]) / f_count
    mal_av = sum(male_day_averages[:max_range]) / m_count
    all_av = sum(day_averages[:max_range]) / a_count

    fem_avs = sorted(female_day_averages[:max_range])
    midpoint = int(len(fem_avs) / 2)
    fem_av_median = [fem_avs[midpoint]]
    mal_avs = sorted(male_day_averages[:max_range])
    midpoint = int(len(mal_avs) / 2)
    mal_av_median = [mal_avs[midpoint]]
    all_avs = sorted(day_averages[:max_range])
    midpoint = int(len(all_avs) / 2)
    all_av_median = [all_avs[midpoint]]

    axes7.bar(1, fem_av, label="female")
    axes7.bar(2, mal_av, label="male")
    axes7.bar(3, all_av, label="all")
    axes7.set_title("Daily average dosages")
    axes7.set_ylabel("Average dosages")
    axes7.set_xlabel("Groups")
    axes7.legend()

    fig8, axes8 = plt.subplots()
    axes8.bar(1, fem_av_median, label="female")
    axes8.bar(2, mal_av_median, label="male")
    axes8.bar(3, all_av_median, label="all")
    axes8.set_title("Median of daily average dosages")
    axes8.set_ylabel("Median of average dosages")
    axes8.set_xlabel("Groups")
    axes8.legend()

    padded_daily_dosages_zipped = zip(*padded_all_dosages)
    padded_daily_dosages_zipped = list(padded_daily_dosages_zipped)

    day_1_prescribs = padded_daily_dosages_zipped[0]
    day_2_prescribs = padded_daily_dosages_zipped[1]
    day_3_prescribs = padded_daily_dosages_zipped[2]
    day_4_prescribs = padded_daily_dosages_zipped[3]
    day_5_prescribs = padded_daily_dosages_zipped[4]

    fig9, axes9 = plt.subplots()
    female_day_vals = []
    female_durations = []
    male_day_vals = []
    male_durations = []
    for i in range(0, len(day_1_prescribs)):
        gender = genders[i]
        if gender == 'F':
            female_day_vals.append(day_1_prescribs[i])
            female_durations.append(admit_durations[i])
        if gender == 'M':
            male_day_vals.append(day_1_prescribs[i])
            male_durations.append(admit_durations[i])

    print("male durations = ", male_durations)
    axes9.scatter(female_durations, female_day_vals, marker="o", color="g", label="female")
    axes9.scatter(male_durations, male_day_vals, marker="o", color="r", label="male")
    axes9.set_xlabel("Admit duration")
    axes9.set_ylabel("Dosage Days")
    axes9.set_title("Day 5 Dosage Days against Admit Duration")
    axes9.legend()

    print("day 1 prescribs = ", day_1_prescribs, len(day_1_prescribs), type(day_1_prescribs),
          (max(day_1_prescribs) - min(day_1_prescribs)))
    # np.random.seed(19680801)
    #
    # mu, sigma = 100, 15
    # x = mu + sigma * np.random.randn(10000)
    # print("x = ", x, len(x), type(x))
    #
    # # the histogram of the data
    # n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    #
    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.xlim(40, 160)
    # plt.ylim(0, 0.03)
    # plt.grid(True)
    # plt.show()

    fig10, axes10 = plt.subplots()
    # n, bins, patches = axes10.hist(day_1_prescribs, 100, density=True, facecolor='g', alpha=0.75)

    sns.kdeplot(female_day_vals, label="female")
    sns.kdeplot(male_day_vals, label="male")
    # sns.kdeplot(day_1_prescribs, label="all")
    axes10.legend()
    axes10.set_title("Day 1 distributions for different genders")
    # axes10.set_xlim(0, 2)

    print("max = ", max(admit_durations), "min = ", min(admit_durations))
    fig11, axes11 = plt.subplots()
    all_averages = []
    female_averages = []
    male_averages = []
    for i in range(0, len(padded_all_dosages)):
        dosages = padded_all_dosages[i][:1]
        admit_duration = admit_durations[i]
        gender = genders[i]
        sum_dosage = sum(dosages)
        average_dosage = sum_dosage  # / (admit_duration + 1)
        all_averages.append(average_dosage)

        if gender == 'F':
            female_averages.append(average_dosage)
        if gender == 'M':
            male_averages.append(average_dosage)

    print("min of all dosages = ", min(all_averages), "max of all dosages = ", max(all_averages))
    print("min of all female dosages = ", min(female_averages), "max of all female dosages = ", max(female_averages))
    print("min of all male dosages = ", min(male_averages), "max of all male dosages = ", max(male_averages))

    sns.kdeplot(all_averages, label="all")
    sns.kdeplot(female_averages, label="female")
    sns.kdeplot(male_averages, label="male")
    # axes11.set_title("Kernel Density Estimation for Average Dosage Days")
    axes11.set_title("Kernel Density Estimation for Dosage Days for Admit Day 1")
    # axes11.set_xlabel("Average Dosage Days")
    axes11.set_xlabel("Dosage Days")
    axes11.legend()
    plt.show()


def assign_narcotic_mme_values():
    # TODO: DELETE this. note: replaced by the assign_mme_values in data processing.
    names = pd.read_csv("./data/analysis-v2/ami-patients-narcotic-prescriptions.csv")["DRUG"].unique().tolist()
    print("names = ", [name.lower() for name in names])

    obj = {
        "fentanyl citrate": 7.2,
        "hydromorphone (dilaudid)": 4,
        "morphine sulfate": 1,
        "hydromorphone": 4,
        "oxycodone (immediate release) ": 1.5,
        "hydrocodone - acetaminophen": 1,
        "oxycodone-acetaminophen": 1.5,
        "oxycodone": 1.5,
        "fentanyl patch": 7.2,
        "meperidine": 0.1,
        "oxycodone sr (oxycontin)": 1.5,
        "acetaminophen w/codeine": 0.15,
        "meperidine pf": 0.1,
        "tramadol (ultram)": 0.1,
        "propoxyphene nap.-apap (n-100)": 0.23,
        "tramadol": 0.1,
        "oxycodone-acetaminophen (5mg-325mg)": 1.5,
        "oxycodone-acetaminophen elixir": 1.5
    }
    return obj


def process_narcotic_prescriptions():
    df = pd.read_csv("./data/analysis-v2/ami-patients-narcotic-prescriptions.csv")
    df.columns = [x.lower() for x in df.columns]
    print("df cols = ", df.columns)

    df_with_mme = assign_mme_values(df)

    df_with_mme = df_with_mme[df_with_mme.columns.drop(list(df_with_mme.filter(regex='unnamed')))]
    print("df with mme head = ", df_with_mme.head(), len(df_with_mme))

    df_with_mme.to_csv("./data/analysis-v2/narcotic-prescriptions-with-mme-values.csv")

    df_with_mme_per_day = compute_dosage_per_day(dosage_df=df_with_mme)
    print("expanded df head = ", df_with_mme_per_day.head(), len(df_with_mme_per_day))

    df_with_mme_per_day.to_csv("./data/analysis-v2/expanded-narcotics-prescriptions-with-mme-values.csv")

    summed_patient_mme_df = sum_mme_per_day(df_with_mme_per_day, "./data/analysis-v2/per-day-summed-mme-values.csv")
    print("summed_patient_mme_df = ", summed_patient_mme_df.head(), len(summed_patient_mme_df))

    # mme_per_admit_day
    patient_info_df = pd.read_csv("./data/analysis-v2/ami-patients-admissions.csv")
    hadms, mme_doses = mme_per_admit_day(patient_info_df=patient_info_df,
                                         patients_total_daily_mme_df=summed_patient_mme_df,
                                         column_name="total_mme")

    # print("mme_doses = ", mme_doses)

    padded_dosages = []
    total_mme_doses = []
    for d in mme_doses:
        padded_dosages.append(pad_list_with_zeros(d, 10))
        total_mme_dose = sum(d)
        print("d = ", d, " and total mme = ", total_mme_dose)
        total_mme_doses.append(round(total_mme_dose, 2))
    print("hadms = ", hadms[0:2])
    print("mme_doses = ", mme_doses[0:2])
    hadm_df = pd.DataFrame(hadms, columns=["hadm_id"])
    mme_doses_df = pd.DataFrame(padded_dosages, columns=["day1", "day2", "day3", "day4", "day5", "day6", "day7",
                                                         "day8", "day9", "day10"])
    total_mme_doses_df = pd.DataFrame(total_mme_doses, columns=["ad-total-mme"])

    print("hadm_df head = ", hadm_df.head())
    print("mme_doses_df head = ", mme_doses_df.head())
    print("total_mme_doses_df head = ", total_mme_doses_df.head())

    merged_df = pd.concat([hadm_df, mme_doses_df, total_mme_doses_df], axis=1)
    print("merged_df head = ", merged_df.head())
    merged_df.to_csv("./data/analysis-v2/per-patient-per-day-mme-values.csv")


def process_non_narcotic_prescriptions():
    pain_meds, narcs, _, _, _, _ = assign_medication_types("./data/analysis-v2/unique-medications-list-ami-patients_SPC.xls")
    non_narcs = set(pain_meds) - set(narcs)

    prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-prescriptions-all.csv")
    non_narc_df = prescriptions_df[prescriptions_df["DRUG"].isin(non_narcs)]
    non_narc_df.columns = [col.lower() for col in non_narc_df.columns]

    non_narc_df.to_csv("./data/analysis-v2/ami-patients-non-narcotic-pain-prescriptions.csv")

    print(non_narc_df.head(), len(non_narc_df))

    nn_df_per_day = compute_dosage_per_day(dosage_df=non_narc_df)
    print("expanded df head = ", nn_df_per_day.head(), len(nn_df_per_day))

    nn_df_per_day.to_csv("./data/analysis-v2/expanded-non-narcotic-prescriptions.csv")

    # sum_non_narcs_per_day
    summed_patient_nn_doses_df = sum_non_narcs_per_day(nn_df_per_day,
                                                       "./data/analysis-v2/per-day-summed-non-narc-dose-values.csv")
    print("summed_patient_nn_doses_df = ", summed_patient_nn_doses_df.head(), len(summed_patient_nn_doses_df))

    patient_info_df = pd.read_csv("./data/analysis-v2/ami-patients-admissions.csv")
    hadms, nn_doses = mme_per_admit_day(patient_info_df=patient_info_df,
                                        patients_total_daily_mme_df=summed_patient_nn_doses_df,
                                        column_name="total_orders")

    padded_dosages = []
    total_doses = []
    for d in nn_doses:
        padded_dosages.append(pad_list_with_zeros(d, 10))
        total_dose = sum(d)
        print("d = ", d, " and total dose = ", total_dose)
        total_doses.append(round(total_dose, 2))
    print("hadms = ", hadms[0:2])
    print("nn_doses = ", nn_doses[0:2])
    hadm_df = pd.DataFrame(hadms, columns=["hadm_id"])
    nn_doses_df = pd.DataFrame(padded_dosages, columns=["day1", "day2", "day3", "day4", "day5", "day6", "day7",
                                                        "day8", "day9", "day10"])
    total_nn_doses_df = pd.DataFrame(total_doses, columns=["ad-total-orders"])

    print("hadm_df head = ", hadm_df.head())
    print("nn_doses_df head = ", nn_doses_df.head())
    print("total_nn_doses_df head = ", total_nn_doses_df.head())

    merged_df = pd.concat([hadm_df, nn_doses_df, total_nn_doses_df], axis=1)
    print("merged_df head = ", merged_df.head())
    merged_df.to_csv("./data/analysis-v2/per-patient-per-day-nn-dose-values.csv")

    # print("non_narcs = ", non_narcs)


def process_aspirin_prescriptions():
    aspirin_prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-aspirin-prescriptions.csv")
    aspirin_prescriptions_df.columns = [col.lower() for col in aspirin_prescriptions_df.columns]
    print(aspirin_prescriptions_df.head(), len(aspirin_prescriptions_df))

    aspirin_df_per_day = compute_dosage_per_day(dosage_df=aspirin_prescriptions_df)
    print("expanded df head = ", aspirin_df_per_day.head(), len(aspirin_df_per_day))

    aspirin_df_per_day.to_csv("./data/analysis-v2/expanded-aspirin-prescriptions.csv")

    summed_patient_aspirin_orders_df = sum_aspirin_per_day(aspirin_df_per_day,
                                                           "./data/analysis-v2/per-day-summed-aspirin-order-values.csv")
    print("summed_patient_aspirin_orders_df = ", summed_patient_aspirin_orders_df.head(),
          len(summed_patient_aspirin_orders_df))

    patient_info_df = pd.read_csv("./data/analysis-v2/ami-patients-admissions.csv")
    hadms, aspirin_orders = mme_per_admit_day(patient_info_df=patient_info_df,
                                              patients_total_daily_mme_df=summed_patient_aspirin_orders_df,
                                              column_name="total_dose")

    padded_dosages = []
    total_doses = []
    for d in aspirin_orders:
        padded_dosages.append(pad_list_with_zeros(d, 10))
        total_dose = sum(d)
        print("d = ", d, " and total dose = ", total_dose)
        total_doses.append(round(total_dose, 2))
    print("hadms = ", hadms[0:2])
    print("nn_doses = ", aspirin_orders[0:2])
    hadm_df = pd.DataFrame(hadms, columns=["hadm_id"])
    aspirin_orders_df = pd.DataFrame(padded_dosages, columns=["day1", "day2", "day3", "day4", "day5", "day6", "day7",
                                                              "day8", "day9", "day10"])
    total_aspirin_orders_df = pd.DataFrame(total_doses, columns=["aspirin-total-dose"])

    print("hadm_df head = ", hadm_df.head())
    print("aspirin_orders_df head = ", aspirin_orders_df.head())
    print("total_aspirin_orders_df head = ", total_aspirin_orders_df.head())
    #
    merged_df = pd.concat([hadm_df, aspirin_orders_df, total_aspirin_orders_df], axis=1)
    print("merged_df head = ", merged_df.head())
    merged_df.to_csv("./data/analysis-v2/per-patient-per-day-aspirin-order-values.csv")


def process_drug_prescriptions(drug_class):
    prescriptions_df = None
    if drug_class == "statin":
        prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-statins-prescriptions.csv")
    elif drug_class == "anti-platelet":
        prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-anti-platelets-prescriptions.csv")
    elif drug_class == "beta-blocker":
        prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-beta-blockers-prescriptions.csv")
    elif drug_class == "ace-inhibitor":
        prescriptions_df = pd.read_csv("./data/analysis-v2/ami-patients-ace-inhibitors-prescriptions.csv")
    prescriptions_df.columns = [col.lower() for col in prescriptions_df.columns]
    print(prescriptions_df.head(), len(prescriptions_df))

    drug_df_per_day = compute_dosage_per_day(dosage_df=prescriptions_df)
    print("expanded df head = ", drug_df_per_day.head(), len(drug_df_per_day))

    drug_df_per_day.to_csv("./data/analysis-v2/expanded-" + drug_class + "-prescriptions.csv")

    summed_patient_drug_orders_df = sum_non_narcs_per_day(drug_df_per_day,
                                                          "./data/analysis-v2/per-day-summed-" + drug_class + "-order-values.csv")
    print("summed_patient_drug_orders_df = ", summed_patient_drug_orders_df.head(),
          len(summed_patient_drug_orders_df))

    patient_info_df = pd.read_csv("./data/analysis-v2/ami-patients-admissions.csv")
    hadms, drug_orders = mme_per_admit_day(patient_info_df=patient_info_df,
                                           patients_total_daily_mme_df=summed_patient_drug_orders_df,
                                           column_name="total_doses")

    padded_dosages = []
    total_doses = []
    for d in drug_orders:
        padded_dosages.append(pad_list_with_zeros(d, 10))
        total_dose = sum(d)
        print("d = ", d, " and total dose = ", total_dose)
        total_doses.append(round(total_dose, 2))
    print("hadms = ", hadms[0:2])
    print("drug_doses = ", drug_orders[0:2])
    hadm_df = pd.DataFrame(hadms, columns=["hadm_id"])
    drug_orders_df = pd.DataFrame(padded_dosages, columns=["day1", "day2", "day3", "day4", "day5", "day6", "day7",
                                                           "day8", "day9", "day10"])
    total_drug_orders_df = pd.DataFrame(total_doses, columns=[drug_class + "-orders"])

    print("hadm_df head = ", hadm_df.head())
    print("drug_orders_df head = ", drug_orders_df.head())
    print("total_drug_orders_df head = ", total_drug_orders_df.head())

    merged_df = pd.concat([hadm_df, drug_orders_df, total_drug_orders_df], axis=1)
    print("merged_df head = ", merged_df.head())
    merged_df.to_csv("./data/analysis-v2/per-patient-per-day-" + drug_class + "-order-values.csv")


def process_race_and_gender_patient_proportions(df_key):
    df = pd.read_csv("./data/analysis-v2/ami_patients_features_master.csv")
    mini_df = df[df[df_key] == 1]
    white_female_count = len(mini_df[(mini_df["ethnicity"] == "white") & (mini_df["gender"] == "f")])
    non_white_female_count = len(mini_df[(mini_df["ethnicity"].isin(["asian",
                                                                     "black/african american",
                                                                     "black/cape verdean",
                                                                     "hispanic or latino",
                                                                     "middle eastern",
                                                                     "multi race ethnicity",
                                                                     "other", "white - brazilian"])) & (
                                                 mini_df["gender"] == "f")])
    white_male_count = len(mini_df[(mini_df["ethnicity"] == "white") & (mini_df["gender"] == "m")])
    non_white_male_count = len(mini_df[(mini_df["ethnicity"].isin(["asian",
                                                                   "black/african american",
                                                                   "black/cape verdean",
                                                                   "hispanic or latino",
                                                                   "middle eastern",
                                                                   "multi race ethnicity",
                                                                   "other", "white - brazilian"])) & (
                                               mini_df["gender"] == "m")])

    # print("white female count = ", white_female_count)
    # print("non-white female count = ", non_white_female_count)
    # print("white male count = ", white_male_count)
    # print("non-white male count = ", non_white_male_count)
    return [white_male_count, non_white_male_count, white_female_count, non_white_female_count]
    # for i in range(0, no_of_days):


def count_patients_in_hospital_per_day_by_gender_and_race(no_of_days=5):
    ls = []
    df = pd.read_csv("./data/analysis-v2/ami_patients_features_master.csv")
    white_male_df = df[(df["ethnicity"] == "white") & (df["gender"] == "m")]
    white_female_df = df[(df["ethnicity"] == "white") & (df["gender"] == "f")]
    non_white_male_df = df[(df["ethnicity"].isin(["asian",
                                                  "black/african american",
                                                  "black/cape verdean",
                                                  "hispanic or latino",
                                                  "middle eastern",
                                                  "multi race ethnicity",
                                                  "other", "white - brazilian"])) & (
                                   df["gender"] == "m")]
    non_white_female_df = df[(df["ethnicity"].isin(["asian",
                                                    "black/african american",
                                                    "black/cape verdean",
                                                    "hispanic or latino",
                                                    "middle eastern",
                                                    "multi race ethnicity",
                                                    "other", "white - brazilian"])) & (
                                     df["gender"] == "f")]
    for i in range(1, no_of_days+1):
        l = []
        white_male_count = len(white_male_df[white_male_df["days-in-hosp"] >= i])
        white_female_count = len(white_female_df[white_female_df["days-in-hosp"] >= i])
        non_white_male_count = len(non_white_male_df[non_white_male_df["days-in-hosp"] >= i])
        non_white_female_count = len(non_white_female_df[non_white_female_df["days-in-hosp"] >= i])
        l.append(white_male_count)
        l.append(non_white_male_count)
        l.append(white_female_count)
        l.append(non_white_female_count)
        ls.append(l)
    return ls


# data = pd.read_csv("./data/prescriptions_with_summed_days_grouped_by_date.csv")
# hadm_ids = data["HADM_ID"].unique() # index

# print (read_results("./data/results-20200109-162522.csv"))
# print read_results("./data/results-20200109-181744.csv")

# read_first_prescriptions("./data/results-20200109-183353.csv")
# read_first_prescriptions("./data/results-20200113-145426.csv")
# read_first_prescriptions("./data/prescriptions_edtime_age<=50_pd=pneumonia_no_pmh.csv")
# read_first_prescriptions("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv")

# read_first_prescriptions("./data/heart_attack_sample_original.csv")
# read_first_prescriptions("./data/heart_attack_all_prescribs.csv")

# assign_dosage_days("./data/prescriptions_edtime_age<=50_pd=pneumonia_no_pmh.csv")
# assign_dosage_days("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv")
# assign_dosage_days("./data/results-20200203-131342.csv") # within first week of being in hospital
# assign_dosage_days("./data/results-20200203-132637.csv") # during ER visit only

# assign_dosage_days("./data/heart_attack_sample_original.csv")
# assign_dosage_days("./data/heart_attack_all_prescribs.csv")
# sum_days("./data/prescriptions_with_dosage_days_computed.csv")
# sum_days_by_date("./data/prescriptions_with_dosage_days_computed.csv")

# compute_chi_square("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv", "is_narcotic", "gender")
# compute_chi_square("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv", "is_narcotic", "ETHNICITY")
# compute_chi_square("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv", "is_narcotic", "INSURANCE")


# compute_chi_square("./data/heart_attack_sample_original.csv", "is_narcotic", "INSURANCE")
# compute_chi_square("./data/heart_attack_sample_original.csv", "is_narcotic", "gender")
# compute_chi_square("./data/heart_attack_sample_original.csv", "is_narcotic", "ETHNICITY")

# compute_point_biserial_corr("./data/prescriptions_with_summed_days.csv",
#                             "sum_dosage_days", "gender_num")

# compute_point_biserial_corr("./data/prescriptions-enriched.csv", "dosage_days", "gender_num")
# compute_point_biserial_corr("./data/features.csv", "average_dosage_days", "gender_num")

# fit_poisson_regression("./data/prescriptions_with_summed_days.csv")

# plot_frequencies("./data/prescriptions_with_summed_days.csv", "gender")
# plot_frequencies("./data/prescriptions_with_summed_days.csv", "INSURANCE")
# plot_frequencies("./data/prescriptions_with_summed_days.csv", "DISCHARGE_LOCATION")
# plot_frequencies("./data/prescriptions_with_summed_days.csv", "ETHNICITY")
# plot_frequencies("./data/prescriptions_with_summed_days.csv", "icd9_code")
# #
# #
# plot_multi_category_frequencies("./data/prescriptions_with_summed_days.csv", "INSURANCE", "gender")
# plot_multi_category_frequencies("./data/prescriptions_with_summed_days.csv", "ETHNICITY", "gender")
# plot_multi_category_frequencies("./data/prescriptions_with_summed_days.csv", "DISCHARGE_LOCATION", "gender")
# #
# #
# plot_outcome_against_features("./data/prescriptions_with_summed_days.csv", "gender", "sum_dosage_days")
# plot_outcome_against_features("./data/prescriptions_with_summed_days.csv", "INSURANCE", "sum_dosage_days")
#
# plot_histograms("./data/features.csv")

# plot_box_plot("./data/features.csv")

# plot_stacked_bar_chart("./data/pneumonia-pain-relief-all-days.csv")

# plot_scatter_dosage_gender_against_duration(0, 1)
# plot_scatter_dosage_gender_against_duration(0, 2)
# plot_scatter_dosage_gender_against_duration(0, 3)
# plot_scatter_dosage_gender_against_duration(0, 4)
# plot_scatter_dosage_gender_against_duration(0, 5)
# plot_scatter_dosage_gender_against_duration(0, 6)
# plot_scatter_dosage_gender_against_duration(0, 7)


# dosage_days_per_admission_period("./data/prescriptions_with_summed_days_grouped_by_date.csv")
# all_dosages, genders, are_narcotics, admit_durations = dosage_days_per_admission_period_2("./data/prescriptions_with_summed_days_grouped_by_date.csv")
# assign_narcotic_mme_values()
# graph(all_dosages, genders, are_narcotics, admit_durations, 30)


# process_narcotic_prescriptions()
# process_non_narcotic_prescriptions()

# process_aspirin_prescriptions()
# process_drug_prescriptions(drug_class="statin")
# process_drug_prescriptions(drug_class="beta-blocker")
# process_drug_prescriptions(drug_class="ace-inhibitor")
# process_drug_prescriptions(drug_class="anti-platelet")

# d = count_patients_in_hospital_per_day_by_gender_and_race()
# print(d)
# TODO: write a good main function
