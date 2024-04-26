import pandas as pd


def compute_lab_test_averages():
    """
    Computes the average troponin, and creatinine values by sex for all admission records.
    Used to populate Table 1 in the manuscript
    :return: nothing. summary files are written in ./results/summaries/
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


def count_patients_in_hospital_per_day_by_gender_and_race(no_of_days=5):
    """
    Counts the number of patients by gender and race (i.e., White-Male, White-Female, Non-White-Male, Non-White-Female)
    admitted for at least N days, for each of the days
    :param no_of_days: number of days for which to compute the numbers
    :return: a list of N (i.e., number of days) lists. each entry in the list is a 4 element list
    [White-Male-Count, White-Female-Count, Non-White-Male-Count, Non-White-Female-Count] of the counts.
    """
    ls = []
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
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


if __name__ == '__main__':
    compute_lab_test_averages()
