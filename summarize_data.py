import pandas as pd


def compute_lab_test_averages():
    """

    :return:
    """
    df = pd.read_csv("./data/processed/simplified-ami-patients-troponin-records.csv")
    f_df = df[df["gender"] == "f"]
    m_df = df[df["gender"] == "m"]

    f_troponin_initial_Is = f_df["initial_trop_I"].dropna()
    f_troponin_peak_Is = f_df["peak_trop_I"].dropna()
    f_troponin_initial_Ts = f_df["initial_trop_T"].dropna()
    f_troponin_peak_Ts = f_df["peak_trop_T"].dropna()

    m_troponin_initial_Is = m_df["initial_trop_I"].dropna()
    m_troponin_peak_Is = m_df["peak_trop_I"].dropna()
    m_troponin_initial_Ts = m_df["initial_trop_T"].dropna()
    m_troponin_peak_Ts = m_df["peak_trop_T"].dropna()

    print("F N tropinin I = ", len(f_troponin_initial_Is), len(f_troponin_peak_Is))
    print("F N tropinin T = ", len(f_troponin_initial_Ts), len(f_troponin_peak_Ts))
    print("M N tropinin I = ", len(m_troponin_initial_Is), len(m_troponin_peak_Is))
    print("M N tropinin T = ", len(m_troponin_initial_Ts), len(m_troponin_peak_Ts))

    f_troponin_I_initial_mean = sum(f_troponin_initial_Is) / len(f_troponin_initial_Is)
    f_troponin_I_peak_mean = sum(f_troponin_peak_Is) / len(f_troponin_peak_Is)
    f_troponin_T_initial_mean = sum(f_troponin_initial_Ts) / len(f_troponin_initial_Ts)
    f_troponin_T_peak_mean = sum(f_troponin_peak_Ts) / len(f_troponin_peak_Ts)

    m_troponin_I_initial_mean = sum(m_troponin_initial_Is) / len(m_troponin_initial_Is)
    m_troponin_I_peak_mean = sum(m_troponin_peak_Is) / len(m_troponin_peak_Is)
    m_troponin_T_initial_mean = sum(m_troponin_initial_Ts) / len(m_troponin_initial_Ts)
    m_troponin_T_peak_mean = sum(m_troponin_peak_Ts) / len(m_troponin_peak_Ts)

    fl = ["f", f_troponin_I_initial_mean, f_troponin_I_peak_mean,
          f_troponin_T_initial_mean, f_troponin_T_peak_mean,
          len(f_troponin_initial_Is), len(f_troponin_peak_Is),
          len(f_troponin_initial_Ts), len(f_troponin_peak_Ts)]
    ml = ["m", m_troponin_I_initial_mean, m_troponin_I_peak_mean,
          m_troponin_T_initial_mean, m_troponin_T_peak_mean,
          len(m_troponin_initial_Is), len(m_troponin_peak_Is),
          len(m_troponin_initial_Ts), len(m_troponin_peak_Ts)]

    l = [fl, ml]
    df = pd.DataFrame(l, columns=["gender",
                                  "av_trop_I_initial", "av_trop_I_peak",
                                  "av_trop_T_initial", "av_trop_T_peak",
                                  "count_trop_I_initial", "count_trop_I_peak",
                                  "count_trop_T_initial", "count_trop_T_peak"])
    df.to_csv("./data/results/summaries/ami-patients-gender-troponin-averages.csv")

    c_df = pd.read_csv("./data/processed/simplified-ami-patients-creatinine-records.csv")
    f_c_df = c_df[c_df["gender"] == "f"]
    m_c_df = c_df[c_df["gender"] == "m"]

    f_creatinine_initial = f_c_df["initial_c"].dropna()
    m_creatinine_initial = m_c_df["initial_c"].dropna()

    f_creatinine_initial_mean = sum(f_creatinine_initial) / len(f_creatinine_initial)
    m_creatinine_initial_mean = sum(m_creatinine_initial) / len(m_creatinine_initial)

    c_fl = ["f", f_creatinine_initial_mean, len(f_creatinine_initial)]
    c_ml = ["m", m_creatinine_initial_mean, len(m_creatinine_initial)]

    c_l = [c_fl, c_ml]
    c_averages_df = pd.DataFrame(c_l, columns=["gender", "av_creatinine_initial", "count_creatinine_initial"])
    c_averages_df.to_csv("./data/results/summaries/ami-patients-gender-creatinine-averages.csv", index=False)

    # shock diagnoses
    shocks_df = pd.read_csv("./data/processed/simplified-ami-patients-shock-diagnoses-records.csv")
    f_shocks_df = shocks_df[shocks_df["gender"] == "f"]
    m_shocks_df = shocks_df[shocks_df["gender"] == "m"]

    f_shock_diagnoses = f_shocks_df["shock_diag"].dropna()
    m_shock_diagnoses = m_shocks_df["shock_diag"].dropna()

    f_shock_diagnoses_perc = round((len(f_shock_diagnoses) / len(f_shocks_df)) * 100, 2)
    m_shock_diagnoses_perc = round((len(m_shock_diagnoses) / len(m_shocks_df)) * 100, 2)

    shocks_fl = ["f", f_shock_diagnoses_perc, len(f_shock_diagnoses)]
    shocks_ml = ["m", m_shock_diagnoses_perc, len(m_shock_diagnoses)]

    shocks_l = [shocks_fl, shocks_ml]
    shocks_percs_df = pd.DataFrame(shocks_l, columns=["gender", "perc_with_diagnoses", "count_with_diagnosis"])
    shocks_percs_df.to_csv("./data/results/summaries/ami-patients-gender-percentage-shock-diagnoses.csv", index=False)

    # cardiogenic shock diagnoses
    c_shocks_df = pd.read_csv("./data/processed/simplified-ami-patients-cardiogenic-shock-diagnoses-records.csv")
    f_c_shocks_df = c_shocks_df[c_shocks_df["gender"] == "f"]
    m_c_shocks_df = c_shocks_df[c_shocks_df["gender"] == "m"]

    f_c_shock_diagnoses = f_c_shocks_df["c_shock_diag"].dropna()
    m_c_shock_diagnoses = m_c_shocks_df["c_shock_diag"].dropna()

    f_c_shock_diagnoses_perc = round((len(f_c_shock_diagnoses) / len(f_c_shocks_df)) * 100, 2)
    m_c_shock_diagnoses_perc = round((len(m_c_shock_diagnoses) / len(m_c_shocks_df)) * 100, 2)

    c_shocks_fl = ["f", f_c_shock_diagnoses_perc, len(f_c_shock_diagnoses)]
    c_shocks_ml = ["m", m_c_shock_diagnoses_perc, len(m_c_shock_diagnoses)]

    c_shocks_l = [c_shocks_fl, c_shocks_ml]
    c_shocks_percs_df = pd.DataFrame(c_shocks_l, columns=["gender", "perc_with_diagnoses", "count_with_diagnosis"])
    c_shocks_percs_df.to_csv("./data/results/summaries/ami-patients-gender-percentage-cardiogenic-shock-diagnoses.csv",
                             index=False)


def patient_admission_statistics():
    df = pd.read_csv("./data/analysis-v2/ami_patients_features_master.csv")
    f_df = df[df["gender"] == "f"]
    m_df = df[df["gender"] == "m"]
    print("length of df = ", len(df))
    admitted_until_day_five = df[df["admit_duration(days)"] <= 5]
    print("length of those admitted for at most 5 days = ", len(admitted_until_day_five))

    print("percentage admitted for at most 5 days = ", round((len(admitted_until_day_five) / len(df)) * 100, 2))

    faudf = admitted_until_day_five[admitted_until_day_five["gender"] == "f"]
    maudf = admitted_until_day_five[admitted_until_day_five["gender"] == "m"]

    print("females admitted until day 5 = ", len(faudf))
    print("males admitted until day 5 = ", len(maudf))

    print("percentage of females admitted until day 5 = ", round((len(faudf) / len(f_df)) * 100, 2))
    print("percentage of males admitted until day 5 = ", round((len(maudf) / len(m_df)) * 100, 2))


def compute_population_percentages(filename, column_name, gender, save_name, in_range=False):
    """

    :param filename:
    :param column_name:
    :param gender:
    :param save_name:
    :param in_range:
    :return:
    """
    df = pd.read_csv(filename)
    g_df = df[df["gender"] == gender]

    if in_range:
        ranges = [15, 20, 30, 40, 50]
        column_value_counts = g_df["age"].groupby(pd.cut(g_df.age, ranges)).count()
    else:
        column_value_counts = g_df[column_name].value_counts()
    column_df = pd.DataFrame({column_name: column_value_counts.index,
                              "counts": column_value_counts.values})
    l = list(column_value_counts)
    column_df["percentages"] = [round(x / sum(l) * 100, 1) for x in l]
    print(column_df.head())

    column_df.to_csv(save_name)


# compute_population_percentages(filename="./data/feature-files/ami_patients_features_master.csv",
#                                column_name="ethnicity", gender="m",
#                                save_name="./data/analysis-v2/demographics-files/male_ethnicity_counts.csv")

# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="ethnicity", gender="f",
#                                save_name="./data/analysis-v2/demographics-files/female_ethnicity_counts.csv")
# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="insurance", gender="m",
#                                save_name="./data/analysis-v2/demographics-files/male_insurance_counts.csv")
# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="insurance", gender="f",
#                                save_name="./data/analysis-v2/demographics-files/female_insurance_counts.csv")

# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="non_stemi?", gender="f",
#                                save_name="./data/analysis-v2/demographics-files/female_nstemi_diagnosis_counts.csv")
# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="non_stemi?", gender="m",
#                                save_name="./data/analysis-v2/demographics-files/male_nstemi_diagnosis_counts.csv")

# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="age", gender="f",
#                                save_name="./data/analysis-v2/demographics-files/female_age_counts.csv", in_range=True)
# compute_population_percentages(filename="./data/analysis-v2/ami_patients_features_master.csv",
#                                column_name="age", gender="m",
#                                save_name="./data/analysis-v2/demographics-files/male_age_counts.csv", in_range=True,)




# compute_lab_test_averages()

# patient_admission_statistics()