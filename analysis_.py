import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
import scipy.stats as stats

import data_processing, utils

all_patients_df = data_processing.read_csv_file("./data/healthy_no_pmh_mi_patients_list.csv")
mi_female_totals, mi_male_totals = utils.male_and_female_population_ami(all_patients_df)


def logit_reg_percentage_prescribed_pain_relief(day_dosage_df, admit_days):
    dosages = data_processing.dosage_per_admit_day(day_dosage_df, "total_dosage_days")
    padded_dosages = []
    for d in dosages:
        padded_dosages.append(data_processing.pad_list_with_zeros(d, admit_days))

    binary_padded_dosages = []
    for d in padded_dosages:
        l = []
        for i in d:
            if i > 0:
                value = 1
            else:
                value = 0
            l.append(value)
        binary_padded_dosages.append(l)

    # print(padded_dosages[0])
    # print(binary_padded_dosages[0])

    day_df = pd.DataFrame(binary_padded_dosages, columns=["day1", "day2", "day3", "day4", "day5"])
    unique_day_dosage_df = pd.DataFrame()
    hadm_ids = day_dosage_df["HADM_ID"].unique().tolist()

    for hadm_id in hadm_ids:
        g = day_dosage_df.where(day_dosage_df["HADM_ID"] == hadm_id)
        g = g[pd.notnull(g["HADM_ID"])]
        unique_day_dosage_df = pd.concat([unique_day_dosage_df, g.head(1)])

    print("unique day dosage len = ", len(unique_day_dosage_df))
    # print(day_df.head())
    # unique_day_dosage_df.join(day_df)

    # print(unique_day_dosage_df.head())
    # print(len(unique_day_dosage_df))
    #
    # print(day_df.head())
    # print(len(day_df))
    # # print(len(unique_day_dosage_df))
    # # print(unique_day_dosage_df.head())
    dat1 = unique_day_dosage_df.reset_index(drop=True)
    dat2 = day_df.reset_index(drop=True)

    dosages_full_df = dat1.join(dat2)
    # print(dosages_full_df.head())
    print(len(dosages_full_df))

    dosages_full_df.to_csv("./assigned_pain_med_or_not_on_admit_days1-5_ami-full-df.csv")
    dosages_full_df[['gender', 'day1', 'day2', 'day3', 'day4', 'day5']].to_csv(
        "./assigned_pain_med_or_not_on_admit_days1-5_ami-days-only.csv")

    day1_df = dosages_full_df[['gender', 'age', 'INSURANCE', 'ETHNICITY', 'day1']]
    day2_df = dosages_full_df[['gender', 'age', 'day2']]
    day3_df = dosages_full_df[['gender', 'age', 'day3']]
    day4_df = dosages_full_df[['gender', 'age', 'day4']]
    day5_df = dosages_full_df[['gender', 'age', 'day5']]
    # print(day1_df.head())

    d1_expr = """day1 ~ gender  + age + INSURANCE """
    # Set up the X and y matrices
    y_train, X_train = dmatrices(d1_expr, day1_df, return_type='dataframe')

    print("X_train = ", X_train.head())
    logistic_reg_training_results = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()

    # Print the training summary.
    print(logistic_reg_training_results.summary())


def non_parametric_h_test_numbers_prescribed_pain_relief(day_dosage_df, max_range):
    men_records = day_dosage_df.where(day_dosage_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = day_dosage_df.where(day_dosage_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosages = data_processing.dosage_per_admit_day(men_records, "total_dosage_days")
    men_padded_dosages = []
    for d in men_dosages:
        men_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    men_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*men_padded_dosages)))

    print("men daily dosages = ", men_daily_dosages)

    women_dosages = data_processing.dosage_per_admit_day(women_records, "total_dosage_days")
    women_padded_dosages = []
    for d in women_dosages:
        women_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    women_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*women_padded_dosages)))
    print("women daily dosages = ", women_daily_dosages)

    men_yes = [x for x in men_daily_dosages]
    men_no = [len(men_records["HADM_ID"].unique().tolist()) - x for x in men_daily_dosages]
    # men_no = [mi_male_totals - x for x in men_daily_dosages]
    women_yes = [x for x in women_daily_dosages]
    women_no = [len(women_records["HADM_ID"].unique().tolist()) - x for x in women_daily_dosages]
    # women_no = [mi_female_totals - x for x in women_daily_dosages]

    print("total no. of men = ", len(men_records["HADM_ID"].unique().tolist()))
    print("total no. of women = ", len(women_records["HADM_ID"].unique().tolist()))

    print("men's = ", men_yes, men_no, sum(men_yes), sum(men_no))
    print("women's = ", women_yes, women_no, sum(women_yes), sum(women_no))

    oddsratio1, pvalue1 = stats.fisher_exact([[men_yes[0], men_no[0]], [women_yes[0], women_no[0]]])
    print(oddsratio1)
    print(pvalue1)

    oddsratio1, pvalue1 = stats.fisher_exact([[men_yes[0], men_no[0]], [women_yes[0], women_no[0]]],
                                             alternative="less")
    print("here ", oddsratio1)
    print("and here ", pvalue1)

    oddsratio2, pvalue2 = stats.fisher_exact([[men_yes[1], men_no[1]], [women_yes[1], women_no[1]]])
    print(oddsratio2)
    print(pvalue2)

    oddsratio2, pvalue2 = stats.fisher_exact([[men_yes[1], men_no[1]], [women_yes[1], women_no[1]]],
                                             alternative="less")
    print("here 2 ", oddsratio2)
    print("and here 2 ", pvalue2)

    oddsratio3, pvalue3 = stats.fisher_exact([[men_yes[2], men_no[2]], [women_yes[2], women_no[2]]])
    print(oddsratio3)
    print(pvalue3)

    oddsratio3, pvalue3 = stats.fisher_exact([[men_yes[2], men_no[2]], [women_yes[2], women_no[2]]],
                                             alternative="less")
    print("here 3 ", oddsratio3)
    print("and here 3 ", pvalue3)

    oddsratio4, pvalue4 = stats.fisher_exact([[men_yes[3], men_no[3]], [women_yes[3], women_no[3]]])
    print(oddsratio4)
    print(pvalue4)

    oddsratio4, pvalue4 = stats.fisher_exact([[men_yes[3], men_no[3]], [women_yes[3], women_no[3]]],
                                             alternative="less")
    print("here 4 ", oddsratio4)
    print("and here 4 ", pvalue4)

    oddsratio5, pvalue5 = stats.fisher_exact([[men_yes[4], men_no[4]], [women_yes[4], women_no[4]]])
    print(oddsratio5)
    print(pvalue5)

    oddsratio5, pvalue5 = stats.fisher_exact([[men_yes[4], men_no[4]], [women_yes[4], women_no[4]]],
                                             alternative="less")
    print("here 5 ", oddsratio5)
    print("and here 5 ", pvalue5)

    oddsratio10, pvalue10 = stats.fisher_exact([[sum(men_yes), sum(men_no)], [sum(women_yes), sum(women_no)]])
    print("10 here OR ", oddsratio10)
    print("10 here PV ", pvalue10)


def linear_regression_average_dosage_days(total_dosage_days_df):  # needs to be renamed
    men_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosage_days = men_records["total_dosage_days"]
    women_dosage_days = women_records["total_dosage_days"]

    men_narc_dosage_days = men_records["narcotic_dosage_days"]
    women_narc_dosage_days = women_records["narcotic_dosage_days"]

    men_non_narc_dosage_days = men_records["non_narcotic_dosage_days"]
    women_non_narc_dosage_days = women_records["non_narcotic_dosage_days"]

    print("men's mean = ", men_dosage_days.mean(axis=0, skipna=True))
    print("women's mean = ", women_dosage_days.mean(axis=0, skipna=True))

    print("men's std = ", men_dosage_days.std(axis=0, skipna=True))
    print("women's std = ", women_dosage_days.std(axis=0, skipna=True))

    print("men's narc mean = ", men_narc_dosage_days.mean(axis=0, skipna=True))
    print("women's narc mean = ", women_narc_dosage_days.mean(axis=0, skipna=True))

    print("men's narc std = ", men_narc_dosage_days.std(axis=0, skipna=True))
    print("women's narc std = ", women_narc_dosage_days.std(axis=0, skipna=True))

    print("men's non-narc mean = ", men_non_narc_dosage_days.mean(axis=0, skipna=True))
    print("women's non-narc mean = ", women_non_narc_dosage_days.mean(axis=0, skipna=True))

    print("men's non-narc std = ", men_non_narc_dosage_days.std(axis=0, skipna=True))
    print("women's non-narc std = ", women_non_narc_dosage_days.std(axis=0, skipna=True))

    # print("men's records means = ", men_records.mean(axis=0, skipna=True))
    # print("women's records means = ", women_records.mean(axis=0, skipna=True))
    #
    # print("men's records std = ", men_records.std(axis=0, skipna=True))
    # print("women's records std = ", women_records.std(axis=0, skipna=True))
    #
    # men_average_total_dosage_days = sum(men_dosage_days) / len(men_dosage_days)
    # women_average_total_dosage_days = sum(women_dosage_days) / len(women_dosage_days)
    # print("checking lens == ", len(men_records), len(men_dosage_days))

    print("len of men's records = ", len(men_records))
    print("len of women's records = ", len(women_records))
    # print("computed means = M: ", men_average_total_dosage_days, " F: ", women_average_total_dosage_days)

    print(stats.ttest_ind(men_dosage_days, women_dosage_days))
    print(stats.ttest_ind(men_narc_dosage_days, women_narc_dosage_days))
    print(stats.ttest_ind(men_non_narc_dosage_days, women_non_narc_dosage_days))


def two_sample_t_test(aspirin_prescribs_df):
    men_records = aspirin_prescribs_df.where(aspirin_prescribs_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = aspirin_prescribs_df.where(aspirin_prescribs_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosage_days = men_records["total_dosage_days"]
    women_dosage_days = women_records["total_dosage_days"]

    print(stats.ttest_ind(men_dosage_days, women_dosage_days))


def non_parametric_h_test_numbers_prescribed_pain_relief_by_race(day_dosage_df, max_range):
    white = ["WHITE"]
    white_records = day_dosage_df[day_dosage_df.ETHNICITY.isin(white)]

    white_and_unspecified = ["WHITE", "UNKNOWN/NOT SPECIFIED"]
    non_white_records = day_dosage_df[~day_dosage_df.ETHNICITY.isin(white_and_unspecified)]

    print("len of white_records 2 = ", len(white_records))
    print(white_records.head())

    print("len of non_white_records  = ", len(non_white_records))
    print(non_white_records.head())

    white_dosages = data_processing.dosage_per_admit_day(white_records, "total_dosage_days")
    white_padded_dosages = []
    for d in white_dosages:
        white_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    white_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*white_padded_dosages)))

    print("white daily dosages = ", white_daily_dosages)

    non_white_dosages = data_processing.dosage_per_admit_day(non_white_records, "total_dosage_days")
    non_white_padded_dosages = []
    for d in non_white_dosages:
        non_white_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    non_white_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero,
                                       zip(*non_white_padded_dosages)))
    print("non_white daily dosages = ", non_white_daily_dosages)

    white_yes = [x for x in white_daily_dosages]
    white_no = [len(white_records["HADM_ID"].unique().tolist()) - x for x in white_daily_dosages]
    non_white_yes = [x for x in non_white_daily_dosages]
    non_white_no = [len(non_white_records["HADM_ID"].unique().tolist()) - x for x in non_white_daily_dosages]

    print("white's = ", white_yes, white_no, sum(white_yes), sum(white_no))
    print("non_white's = ", non_white_yes, non_white_no, sum(non_white_yes), sum(non_white_no))

    print("white no. = ", len(white_records["HADM_ID"].unique().tolist()))
    print("non-white no. = ", len(non_white_records["HADM_ID"].unique().tolist()))

    oddsratio1, pvalue1 = stats.fisher_exact([[white_yes[0], white_no[0]], [non_white_yes[0], non_white_no[0]]])
    print(oddsratio1)
    print(pvalue1)

    oddsratio1, pvalue1 = stats.fisher_exact([[white_yes[0], white_no[0]], [non_white_yes[0], non_white_no[0]]],
                                             alternative="less")
    print("here ", oddsratio1)
    print("and here ", pvalue1)

    oddsratio10, pvalue10 = stats.fisher_exact(
        [[sum(white_yes), sum(white_no)], [sum(non_white_yes), sum(non_white_no)]])
    print("10 here OR ", oddsratio10)
    print("10 here PV ", pvalue10)


def t_test_medication_dosage(df_key, key_name, sensitive_attribute):
    df = pd.read_csv("./data/analysis-v2/ami_patients_features_master.csv")
    non_white_ethnicities = ["asian",
                             "black/african american",
                             "black/cape verdean",
                             "hispanic or latino",
                             "middle eastern",
                             "multi race ethnicity",
                             "other", "white - brazilian"]
    if sensitive_attribute == "gender":
        minority_df = df[df["gender"] == "f"]
        majority_df = df[df["gender"] == "m"]
    elif sensitive_attribute == "race":
        majority_df = df[df["ethnicity"] == "white"]
        minority_df = df[df["ethnicity"].isin(non_white_ethnicities)]
    else:
        majority_df = None
        minority_df = None
    majority_values = majority_df[df_key].dropna()
    minority_values = minority_df[df_key].dropna()

    # TODO: confirm this average is correct
    # TODO: Confirm whether average should be over entire df or just those who received the treatment

    majority_average = sum(majority_values) / len(majority_values)
    minority_average = sum(minority_values) / len(minority_values)

    test_val = stats.ttest_ind(majority_values, minority_values)
    statistic = list(test_val)[0]
    p_value = list(test_val)[1]
    print("test_val = ", test_val, type(test_val), list(test_val))
    print("statistic = ", statistic, " p-value = ", p_value)
    l = [key_name, sensitive_attribute, p_value, statistic, majority_average, minority_average]
    return l


def process_t_test_results(sensitive_attributes, keys, key_names, write_file_name):
    ls = []
    col_names = ["feature-name", "sensitive-attribute", "p-value", "statistic", "maj_average_val", "min_average_val"]

    for sa in sensitive_attributes:
        for i in range(len(keys)):
            key = keys[i]
            key_name = key_names[i]
            l = t_test_medication_dosage(df_key=key, key_name=key_name, sensitive_attribute=sa)
            ls.append(l)
    df = pd.DataFrame(ls, columns=col_names)
    print("df head = ", df.head())
    df.to_csv(write_file_name)


# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_pain_prescribs-with-aspirin.csv")
# df = data_processing.assign_is_narcotic_value(data)
# df = data_processing.assign_dosage_days(df)
# gf = data_processing.compute_dosage_per_day(df)
# hf = data_processing.sum_dosage_per_day(gf)
#
# # logit_reg_percentage_prescribed_pain_relief(hf, 5)
#
# non_parametric_h_test_numbers_prescribed_pain_relief(hf, 5)

# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# linear_regression_average_dosage_days(mf)

# # logit_reg_percentage_prescribed_pain_relief(hf, 5)

# non_parametric_h_test_numbers_prescribed_pain_relief(hf, 5)
# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# two_sample_t_test(mf)

# non_parametric_h_test_numbers_prescribed_pain_relief_by_race(hf, 5)


#  creating data files on dosage days per hadm_id
# 1. aspirin
# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_aspirin_prescribs.csv")
# df = data_processing.assign_is_narcotic_value(data)
#
# # df.to_csv("./data/output_files/aspirin/prescriptions_assigned_narc_or_non_narc_value.csv")
#
# df = data_processing.assign_dosage_days(df)
#
# # df.to_csv("./data/output_files/aspirin/prescriptions_assigned_dosage_days.csv")
# gf = data_processing.compute_dosage_per_day(df)
#
# # gf.to_csv("./data/output_files/aspirin/prescriptions_with_dosage_computed_per_day.csv")
# hf = data_processing.sum_dosage_per_day(gf)
# # hf.to_csv("./data/output_files/aspirin/prescriptions_with_dosage_days_summed_per_day.csv")
#
# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# l = ["HADM_ID", "age", "gender", "gender_num", "total_dosage_days"]
#
# mmf = mf[l]
# # mf.to_csv("./data/output_files/aspirin/prescriptions_with_total_dosage_days_per_hadm_id.csv")
# mmf.to_csv("./data/output_files/aspirin/aspirin_total_dosages.csv")
#
# # 2. all pain relief prescriptions TODO: if aspirin needed to be analyzed separately, use the pain_data only below
# # pain_data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_pain_prescribs-with-aspirin.csv")
# df = data_processing.assign_is_narcotic_value(data)
#
# # df.to_csv("./data/output_files/all/prescriptions_assigned_narc_or_non_narc_value.csv")
#
# df = data_processing.assign_dosage_days(df)
#
# # df.to_csv("./data/output_files/all/prescriptions_assigned_dosage_days.csv")
# gf = data_processing.compute_dosage_per_day(df)
#
# # gf.to_csv("./data/output_files/all/prescriptions_with_dosage_computed_per_day.csv")
# hf = data_processing.sum_dosage_per_day(gf)
# # hf.to_csv("./data/output_files/all/prescriptions_with_dosage_days_summed_per_day.csv")
#
# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# l = ["HADM_ID", "age", "gender", "gender_num", "total_dosage_days"]
#
# mmf = mf[l]
# # mf.to_csv("./data/output_files/all/prescriptions_with_total_dosage_days_per_hadm_id.csv")
# mmf.to_csv("./data/output_files/all/all_pain_medication_total_dosages.csv")

# t_test_medication_dosage(df_key="ad-average-mme", sensitive_attribute="race", key_name="average-mme")

attributes = ["gender", "race"]
keys = ["ad-average-mme", "ad-average-nnd", "ad-average-asd", "ad-average-statin",
        "ad-average-anti-platelet", "ad-average-ace-inhibitor", "ad-average-beta-blocker"]
key_names = ["average-mme", "average-non-narc-orders", "average-aspirin-orders", "average-statin-orders",
             "average-anti-platelet-orders", "average-ace-inhibitor-orders", "average-beta-blocker-orders"]

process_t_test_results(sensitive_attributes=attributes, keys=keys, key_names=key_names,
                       write_file_name="./data/analysis-v2/h-tests/t-tests/t-test-pvalues.csv")
