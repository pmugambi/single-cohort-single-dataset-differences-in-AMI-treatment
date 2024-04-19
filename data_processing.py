from __future__ import division
import pandas as pd


def read_csv_file(filename):
    """
    Reads a CSV file using pandas
    :param filename: THe CSV file name
    :return: A DF of the read data
    """
    return pd.read_csv(filename)


def read_column_values(df, column_name):
    """
    Reads the values of the specified column <column_name> in the DF <df>
    :param df: A DF
    :param column_name: Column name of the column to be read
    :return: Values of the column <column_name> in the DF <df?
    """
    return df[column_name]


# def compute_percentage_received_treatment(treatment_df, treatment_col_name, base_col_name,
#                                           female_count_col_name, male_count_col_name,
#                                           female_population_count, male_population_count,
#                                           treatment_type, extra_col=None):
#     """
#     Compute the percentage of patients who received a certain service
#     :param treatment_df: the DF of the treatment values
#     :param treatment_col_name: the treatment column name in the DF
#     :param base_col_name: the column name in the DF that is always filled and can be used to filter the DF
#     :param female_count_col_name: the column in the DF containing count of female patients that received the treatment
#     :param male_count_col_name: the column in the DF containing count of male patients that received the treatment
#     :param female_population_count: the total number of female patients in the sample
#     :param male_population_count: the total number of male patients in the sample
#     :return: a list of objects containing the treatment id, treatment name, and percentage of patients who received
#      that treatment grouped by gender - for each treatment in the treatment Df
#     """
#     # df = treatment_df.where(9 < treatment_df["hadm_unique"] < 260)
#     print("len of original df = ", len(treatment_df))
#     # df = treatment_df.where(treatment_df['hadm_unique'].between(9, 260, inclusive=False))
#     # df = df[pd.notnull(df[base_col_name])]
#
#     # print("len of reduced df = ", len(df))
#     # treatments_ids_list = list(read_column_values(df, treatment_col_name).unique())
#     objs = []
#     # for treatment_id in treatments_ids_list:
#     df = pd.DataFrame()
#     names = ["treatment_id", "treatment_name", "total_patients", "male_perc", "female_perc"]
#     names1 = ["treatment_id", "treatment_name", extra_col, "total_patients", "male_perc", "female_perc"]
#     # df = pd.concat([df, pd.DataFrame(names)])
#
#     print("extra col name = ", extra_col)
#     for i in range(0, len(treatment_df)):
#         obj = {}
#         row = treatment_df.iloc[[i]]
#
#         treatment_id = row[treatment_col_name]
#         female_count = row[female_count_col_name]
#         male_count = row[male_count_col_name]
#
#         female_perc = (float(female_count) / female_population_count) * 100
#         male_perc = (float(male_count) / male_population_count) * 100
#
#         obj["treatment_id"] = list(treatment_id)[0]
#         obj["treatment_name"] = list(row[base_col_name])[0]  # check if this is the case for all treatment types,
#         # i.e. that the base_col_name can be picked to be the treatment name
#
#         if extra_col is not None:
#             obj[extra_col] = list(row[extra_col])[0]
#
#         obj["total_patients"] = list(row["hadm_unique"])[0]
#         obj["male_perc"] = male_perc
#         obj["female_perc"] = female_perc
#         # print(obj)
#         objs.append(obj)
#         try:
#             t_df = pd.DataFrame([[list(treatment_id)[0], list(row[base_col_name])[0], list(row[extra_col])[0],
#                                   list(row["hadm_unique"])[0], male_perc, female_perc]], columns=names1)
#         except Exception:
#             t_df = pd.DataFrame([[list(treatment_id)[0], list(row[base_col_name])[0],
#                                   list(row["hadm_unique"])[0], male_perc, female_perc]], columns=names)
#         df = pd.concat([df, t_df])
#     print(df.head())
#     df.to_csv("./" + treatment_type + "_percentages_by_gender.csv")
#     return objs


def organize_patient_procedures():
    """

    :return:
    """
    procedures = pd.read_csv("./data/raw/procedures_on_healthy_ami_patients.csv")
    weights = pd.read_csv("./data/raw/weight_records_ami_patients.csv")
    hadm_ids = procedures['HADM_ID'].unique()

    print("hadm_ids = ", len(hadm_ids))

    headers = ["HADM_ID", "age", "gender", "icd9_code", "seq_num", "short_title", "long_title"]
    final_df = pd.DataFrame()
    for hadm_id in hadm_ids:
        # get records for that hadm_id
        hadm_procedures = procedures[procedures["HADM_ID"] == hadm_id][headers].sort_values(
            by="seq_num", axis=0)
        hadm_weights = weights[weights["HADM_ID"] == hadm_id][["VALUE", "VALUEUOM", "CHARTTIME", "label"]].sort_values(
            by="CHARTTIME", axis=0)
        print("hadm_id = ", hadm_id, " len of hadm_procedures = ", len(hadm_procedures))
        # print("hadm_procedures for hadm_id ", hadm_id, " = ", hadm_procedures)
        gender = hadm_procedures['gender'].tolist()[0]
        age = hadm_procedures['age'].tolist()[0]

        # print("weight df = ", hadm_weights["VALUE"])

        weight_list = hadm_weights['VALUE'].dropna().tolist()

        # assigning a null value for patients without a recorded weight
        if len(weight_list) > 0:
            weight = round(weight_list[0], 2)
        else:
            weight = None

        # In future, it may be worth checking unit of measurement (UOM).
        # For now, all weights are in kgs so I'll ignore this step
        weightuom_list = hadm_weights['VALUEUOM'].tolist()

        print("gender = ", gender, " and age = ", age)
        print("weight = ", weight)
        procedures_icd = hadm_procedures['icd9_code'].tolist()
        procedures_short_title = hadm_procedures['short_title'].tolist()
        procedures_long_title = hadm_procedures['long_title'].tolist()

        num_of_procedures = len(hadm_procedures)

        print("procedures icd codes = ", procedures_icd)
        df = pd.DataFrame([[hadm_id, gender, age, weight, num_of_procedures, procedures_icd,
                            procedures_short_title,
                            procedures_long_title]],
                          columns=["hadm_id", "gender", "age", "weight(kg)", "p_count", "icd9_codes", "short_titles",
                                   "long_titles"])
        # print("df = ", df)
        final_df = pd.concat([final_df, df])
    print("final df = ", final_df)
    final_df.to_csv("./data/processed/organized_procedures_for_ami_patients.csv")


# def create_patient_statistics_file():
#     ami_patients = read_csv_file("./data/healthy_no_pmh_mi_patients_list.csv")[["HADM_ID", "ADMITTIME", "DISCHTIME",
#                                                                                 "age", "gender"]]
#     print("ami_patients.head() = ", ami_patients.head(), len(ami_patients))
#     aspirin_meds = read_csv_file("./data/output_files/aspirin/aspirin_total_dosages.csv")[["HADM_ID",
#                                                                                            "total_dosage_days"]]
#     print("aspirin_meds.head() = ", aspirin_meds.head(), len(aspirin_meds))
#
#     df1 = pd.merge(ami_patients, aspirin_meds, how="outer", on="HADM_ID")
#     df1.columns = ["HADM_ID", "admittime", "dischtime", "age", "gender", "aspirin_dd"]
#
#     print("df1.head() = ", df1.head(), len(df1))
#     all_pain_meds = read_csv_file("./data/output_files/all/all_pain_medication_total_dosages.csv")[["HADM_ID",
#                                                                                                     "total_dosage_days"]]
#
#     df2 = pd.merge(df1, all_pain_meds, how="outer", on="HADM_ID")
#     df2.columns = ["hadm_id", "admittime", "dischtime", "age", "gender", "aspirin_dd", "pain_med_dd"]
#     print("df2.head() = ", df2.head(), len(df2))
#     procedures = read_csv_file("./data/organized_procedures_for_ami_patients.csv")[["hadm_id", "weight(kg)", "p_count"]]
#
#     df3 = pd.merge(df2, procedures, how="outer", on="hadm_id")
#     df3.columns = ["hadm_id", "admittime", "dischtime", "age", "gender", "aspirin_dd", "pain_med_dd", "weight(kg)",
#                    "p_count"]
#     print("df3.head() = ", df3.head(), len(df3))
#
#     df3["aspirin_dd"] = df3["aspirin_dd"].fillna(0)
#     df3["pain_med_dd"] = df3["pain_med_dd"].fillna(0)
#
#     df3['dischtime'] = pd.to_datetime(df3['dischtime'])
#     df3['admittime'] = pd.to_datetime(df3['admittime'])
#
#     admit_duration = (df3['dischtime'] - df3['admittime']).dt.days
#     df3['admit_days'] = admit_duration
#
#     print("df3.head() = ", df3.head(), len(df3))
#     df3 = df3[["hadm_id", "age", "gender", "weight(kg)", "admit_days", "aspirin_dd", "pain_med_dd", "p_count"]]
#     print("df3.head() = ", df3.head(), len(df3))
#     df3["received_pain_med"] = 0
#     df3.loc[df3.pain_med_dd > 0, ['received_pain_med']] = 1
#     df3["received_aspirin"] = 0
#     df3.loc[df3.aspirin_dd > 0, ['received_aspirin']] = 1
#     df3.to_csv("./data/output_files/ami_patient_statistics.csv")
#     df3["gender_num"] = 0
#     df3.loc[df3.gender == 'M', ['gender_num']] = 1
#     df3 = df3[["age", "gender_num", "weight(kg)", "admit_days", "aspirin_dd", "pain_med_dd", "p_count",
#                "received_pain_med", "received_aspirin"]]
#     print("df3.head() = ", df3.head(), len(df3))
#     df3.to_csv("./data/output_files/ami_patient_statistics_without_hadm_id.csv")


def create_patient_features_file():
    """

    :return: nothing. the features are written into a patients'-file and saved in ./data/feature-files
    """
    info_df = read_csv_file("./data/raw/ami-patients-admissions.csv")[["SUBJECT_ID", "HADM_ID",
                                                                       "INSURANCE", "ETHNICITY",
                                                                       "gender", "age", "admit_duration",
                                                                       "HOSPITAL_EXPIRE_FLAG"]]
    column_names = ["subject_id", "hadm_id", "insurance", "ethnicity", "gender", "age", "admit_duration(days)",
                    "died?"]
    info_df.columns = column_names

    info_df["insurance"] = info_df["insurance"].str.lower()
    info_df["ethnicity"] = info_df["ethnicity"].str.lower()
    info_df["gender"] = info_df["gender"].str.lower()

    diagnoses_df = read_csv_file("./data/raw/ami-patients-primary-diagnoses.csv")[["HADM_ID", "ICD9_CODE"]]
    diagnoses_df.columns = ["hadm_id", "icd9_code"]

    df = pd.merge(info_df, diagnoses_df, how="outer", on="hadm_id")

    df["non_stemi?"] = 0
    df.loc[df.icd9_code == 41071, "non_stemi?"] = 1

    shock_df = read_csv_file("./data/raw/ami-patients-shock-diagnoses-records.csv")[["HADM_ID", "ICD9_CODE",
                                                                                             "gender"]]
    shock_df.columns = ["hadm_id", "icd9_code", "gender"]

    df["shock?"] = 0
    ids_with_shock = shock_df["hadm_id"].unique().tolist()
    df.loc[df.hadm_id.isin(ids_with_shock), "shock?"] = 1

    df["c-shock?"] = 0
    c_shock_df = shock_df[shock_df["icd9_code"] == 78551]
    ids_with_c_shock = c_shock_df["hadm_id"].unique().tolist()
    df.loc[df.hadm_id.isin(ids_with_c_shock), "c-shock?"] = 1

    df["pca?"] = 0
    ids_with_pca = read_csv_file("./data/raw/ami-patients-pca-records.csv")["HADM_ID"].unique().tolist()
    print("pca_ids = ", ids_with_pca)

    df.loc[df.hadm_id.isin(ids_with_pca), "pca?"] = 1

    ids_with_aspirin_prescriptions = read_csv_file("./data/raw/ami-patients-aspirin-prescriptions.csv")[
        "HADM_ID"].unique().tolist()
    df["aspirin?"] = 0
    df.loc[df.hadm_id.isin(ids_with_aspirin_prescriptions), "aspirin?"] = 1

    ids_with_liver_conditions = read_csv_file("./data/raw/ami-patients-with-liver-related-diagnoses.csv")[
        "HADM_ID"].unique().tolist()
    df["liver_conditions?"] = 0
    df.loc[df.hadm_id.isin(ids_with_liver_conditions), "liver_conditions?"] = 1

    ids_with_kidney_conditions = read_csv_file("./data/raw/ami-patients-with-kidney-related-diagnoses.csv")[
        "HADM_ID"].unique().tolist()
    print("ids with kidney issues = ", sorted(ids_with_kidney_conditions), len(ids_with_kidney_conditions))
    df["kidney_conditions?"] = 0
    df.loc[df.hadm_id.isin(ids_with_kidney_conditions), "kidney_conditions?"] = 1

    # ids received pain medication
    all_pain_meds, narcotics, _, _, _, _ = assign_medication_types(
        "./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    prescriptions_df = read_csv_file("./data/raw/ami-patients-prescriptions-all.csv")
    ids_with_pain_med_prescriptions = prescriptions_df[
        prescriptions_df["DRUG"].isin(all_pain_meds)]["HADM_ID"].unique().tolist()

    print("ids with pain med prescriptions = ", ids_with_pain_med_prescriptions)
    df["received_pain_med?"] = 0
    df.loc[df.hadm_id.isin(ids_with_pain_med_prescriptions), "received_pain_med?"] = 1

    # # ids received narcotics
    # save narcotics prescriptions file
    narcotics_df = prescriptions_df[prescriptions_df["DRUG"].isin(narcotics)]
    narcotics_df.to_csv("./data/processed/ami-patients-narcotic-prescriptions.csv")

    ids_with_narcotic_prescriptions = prescriptions_df[
        prescriptions_df["DRUG"].isin(narcotics)]["HADM_ID"].unique().tolist()
    print("ids with narcotic prescriptions = ", ids_with_narcotic_prescriptions)
    df["received_narcotic?"] = 0
    df.loc[df.hadm_id.isin(ids_with_narcotic_prescriptions), "received_narcotic?"] = 1

    # ids received procedures? look at stephanie's list to see which procedures to eliminate from procedures list
    procedures_df = read_csv_file("./data/raw/ami-patients-procedures-all.csv")
    surgical, non_surgical, other = assign_procedure_type("./data/annotated/unique-procedures-list-ami-patients.xls")
    surgical_procedures_df = procedures_df[procedures_df["long_title"].isin(surgical)]
    non_surgical_procedures_df = procedures_df[procedures_df["long_title"].isin(non_surgical)]
    other_procedures_df = procedures_df[procedures_df["long_title"].isin(other)]
    print("surgical_procedures_df.head() = ", surgical_procedures_df.head())

    non_other_procedures_df = pd.concat([surgical_procedures_df, non_surgical_procedures_df])
    print("non_other_procedures_df.head() = ", non_other_procedures_df.head())
    ids_received_procedures = non_other_procedures_df["hadm_id"].unique().tolist()

    df["received_procedure?"] = 0
    df.loc[df.hadm_id.isin(ids_received_procedures), "received_procedure?"] = 1

    print("ids received_procedures = ", sorted(ids_received_procedures), len(ids_received_procedures))

    # ids received surgical procedures
    ids_received_surgical_procedures = surgical_procedures_df["hadm_id"].unique().tolist()
    ids_received_non_surgical_procedures = non_surgical_procedures_df["hadm_id"].unique().tolist()
    ids_received_other_procedures = other_procedures_df["hadm_id"].unique().tolist()

    df["surgical_procedure?"] = 0
    df.loc[df.hadm_id.isin(ids_received_surgical_procedures), "surgical_procedure?"] = 1

    df["non_surgical_procedure?"] = 0
    df.loc[df.hadm_id.isin(ids_received_non_surgical_procedures), "non_surgical_procedure?"] = 1

    df["other_procedure?"] = 0
    df.loc[df.hadm_id.isin(ids_received_other_procedures), "other_procedure?"] = 1

    print("ids received surgical procedures = ", ids_received_surgical_procedures,
          len(ids_received_surgical_procedures))
    df.to_csv("./data/feature-files/ami_patients_features_master.csv")


def assign_procedure_type(procedures_file):
    """

    :param procedures_file:
    :return:
    """
    procedures_df = pd.read_excel(procedures_file, header=None, usecols=[0, 1])
    # print(procedures_df.head())
    # print(procedures_df[1])

    surgical_procedures = procedures_df[procedures_df[1] == "Surgical "][0].tolist()
    non_surgical_procedures = procedures_df[procedures_df[1] == "Non-surgical"][0].tolist()
    other_procedures = procedures_df[procedures_df[1] == "Other"][0].tolist()

    # print("surgical procs = ", surgical_procedures, len(surgical_procedures))
    # print("non-surgical procs = ", non_surgical_procedures, len(non_surgical_procedures))
    # print("other procs = ", other_procedures, len(other_procedures))
    return surgical_procedures, non_surgical_procedures, other_procedures


def assign_medication_types(medications_file):
    """
    This function is used in the function below to mark patients who received pain medication,
     narcotic, etc..
    It takes as input an excel file containing names of unique drugs prescribed to the study cohort, then creates
    drug-type groups and returns them.
    :param medications_file: an excel file containing names of unique drugs prescribed to the study cohort
    :return: drug-type groups
    """
    #
    medications_df = pd.read_excel(medications_file, header=None)
    pain_medications_df = medications_df[medications_df[3].isin(["YES", "MAYBE"])]
    pain_meds = pain_medications_df[0].unique().tolist()
    narcotics = pain_medications_df[pain_medications_df[4] == "YES"][0].unique().tolist()
    # non_narcotics = pain_medications_df[pain_medications_df[4] == "NO"][0].unique().tolist()

    # aspirin_df = medications_df[medications_df[6].isin(["YES"])]
    # aspirin_meds = aspirin_df[0].unique().tolist()

    print("narcotics = ", narcotics)
    # print("non-narcotics = ", non_narcotics)
    beta_blockers = medications_df[medications_df[7] == "YES"][0].unique().tolist()
    statins = medications_df[medications_df[8] == "YES"][0].unique().tolist()
    anti_platelets = medications_df[medications_df[9] == "YES"][0].unique().tolist()
    ace_inhibitors = medications_df[medications_df[10] == "YES"][0].unique().tolist()
    return pain_meds, narcotics, beta_blockers, statins, anti_platelets, ace_inhibitors


def write_prescriptions_files(medications_file):
    """
    Given a unique list of drugs, this function uses assign_medication_types() to obtain drug classes,
    and then reads the cohort prescriptions file, and subdivides it into the various drug prescription files
    :param medications_file: an excel file containing names of unique drugs prescribed to the study cohort
    :return: nothing. the created files are written to /data/processed/ folder
    """
    _, _, beta_blockers, statins, anti_platelets, ace_inhibitors = \
        assign_medication_types(medications_file)
    prescriptions_df = pd.read_csv("./data/raw/ami-patients-prescriptions-all.csv")
    # print("beta blockers = ", beta_blockers)
    beta_blockers_df = prescriptions_df[prescriptions_df["DRUG"].isin(beta_blockers)]
    # print("beta blockers df head = ", beta_blockers_df.head())
    statin_df = prescriptions_df[prescriptions_df["DRUG"].isin(statins)]
    anti_platelets_df = prescriptions_df[prescriptions_df["DRUG"].isin(anti_platelets)]
    ace_inhibitors_df = prescriptions_df[prescriptions_df["DRUG"].isin(ace_inhibitors)]

    # save files to prescriptions file
    beta_blockers_df.to_csv("./data/processed/ami-patients-beta-blockers-prescriptions.csv")
    statin_df.to_csv("./data/processed/ami-patients-statins-prescriptions.csv")
    anti_platelets_df.to_csv("./data/processed/ami-patients-anti-platelets-prescriptions.csv")
    ace_inhibitors_df.to_csv("./data/processed/ami-patients-ace-inhibitors-prescriptions.csv")


def clean_troponin_file():
    """

    :return:
    """
    df = pd.read_csv("./data/raw/ami-patients-troponin-records.csv")
    values = df["value"]
    values = values.replace("GREATER THAN 50.0", 50.0)
    values = values.replace("GREATER THAN 50", 50.0)
    values = values.replace("GREATER THAN 25", 25.0)
    values = values.replace("GREATER THAN 25.0", 25.0)
    values = values.replace("GREATER THAN 50 NG/ML", 50.0)
    values = values.replace("LESS THAN 0.01", 0.01)
    values = values.replace("GREATER THAN FIFTY, DILUTIONS PERFORMED UPON REQUEST", 50.0)
    values = values.replace("<0.01", 0.01)
    values = values.replace("<0.3", 0.3)
    values = values.replace(">50", 50.0)
    values = values.replace(">50.0", 50.0)
    print("values = ", list(values))
    values = [float(x) for x in values]
    print("values = ", list(values))

    df["values_mod"] = values
    df.to_csv("./data/processed/ami-patients-troponin-records-modified.csv")


def process_disease_severity_markers():
    """

    :return:
    """
    patients_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
    troponin_df = pd.read_csv("./data/processed/ami-patients-troponin-records-modified.csv")
    creatinine_df = pd.read_csv("./data/raw/ami-patients-creatinine-records.csv")
    creatinine_only_df = creatinine_df[creatinine_df["itemid"] == 50912]

    # 78551
    shock_diagnoses_df = pd.read_csv("./data/raw/ami-patients-shock-diagnoses-records.csv")

    troponin_I_df = troponin_df[troponin_df["label"].str.lower() == "troponin i"]
    troponin_T_df = troponin_df[troponin_df["label"].str.lower() == "troponin t"]

    patients_df.columns = [x.lower() for x in patients_df.columns]
    troponin_df.columns = [x.lower() for x in troponin_df.columns]
    troponin_I_df.columns = [x.lower() for x in troponin_I_df.columns]
    troponin_T_df.columns = [x.lower() for x in troponin_T_df.columns]
    creatinine_only_df.columns = [x.lower() for x in creatinine_only_df.columns]
    shock_diagnoses_df.columns = [x.lower() for x in shock_diagnoses_df.columns]

    cardiogenic_shocks_df = shock_diagnoses_df[shock_diagnoses_df["icd9_code"] == 78551]

    hadms = patients_df["hadm_id"].tolist()
    print("hadms = ", hadms, len(hadms))

    # group troponin records by hadm_id to obtain initial anf peak values

    trop_ls = []
    creat_ls = []
    shock_ls = []
    cardiogenic_shock_ls = []

    for hadm in hadms:
        hadm_troponin_I_df = troponin_I_df[troponin_I_df["hadm_id"] == hadm]
        hadm_troponin_T_df = troponin_T_df[troponin_T_df["hadm_id"] == hadm]
        hadm_creatinine_only_df = creatinine_only_df[creatinine_only_df["hadm_id"] == hadm]
        hadm_shock_df = shock_diagnoses_df[shock_diagnoses_df["hadm_id"] == hadm]
        hadm_cardiogenic_shock_df = cardiogenic_shocks_df[cardiogenic_shocks_df["hadm_id"] == hadm]
        # print("hadm_troponin_records = ", hadm_troponin_df, len(hadm_troponin_df))

        gender = patients_df[patients_df["hadm_id"] == hadm]["gender"].tolist()[0]
        gender = gender.lower()
        print("hadm_id = ", hadm, " and gender = ", gender)
        initial_I = None
        peak_I = None
        initial_T = None
        peak_T = None
        initial_c = None
        shock_diagnosis = None
        cardiogenic_shock_diagnosis = None

        if len(hadm_troponin_I_df) > 0:
            # sort df by date, i.e. charttime, asc
            sorted_hadm_troponin_I_df = hadm_troponin_I_df.sort_values(by="charttime",
                                                                       ascending=True)
            initial_I = sorted_hadm_troponin_I_df.head(1)["values_mod"].tolist()[0]
            peak_I = max(sorted_hadm_troponin_I_df["values_mod"])
            print("hadm_id = ", hadm, " initial reading = ", initial_I, " peak = ", peak_I)

        if len(hadm_troponin_T_df) > 0:
            # sort df by date, i.e. charttime, asc
            sorted_hadm_troponin_T_df = hadm_troponin_T_df.sort_values(by="charttime",
                                                                       ascending=True)
            initial_T = sorted_hadm_troponin_T_df.head(1)["values_mod"].tolist()[0]
            peak_T = max(sorted_hadm_troponin_T_df["values_mod"])
            print("hadm_id = ", hadm, " initial reading = ", initial_T, " peak = ", peak_T)

        # process creatinine averages here
        if len(hadm_creatinine_only_df) > 0:
            # sort df by date, i.e. charttime, asc
            sorted_hadm_creatinine_only_df = hadm_creatinine_only_df.sort_values(by="charttime",
                                                                                 ascending=True)
            initial_c = sorted_hadm_creatinine_only_df.head(1)["value"].tolist()[0]

        if len(hadm_shock_df) > 0:
            # should ideally be one diagnosis per person, whether they got one or not, right?
            # should a repeat diagnosis really matter?
            shock_diagnosis = 1

        if len(hadm_cardiogenic_shock_df) > 0:
            # should ideally be one diagnosis per person, whether they got one or not, right?
            # should a repeat diagnosis really matter?
            cardiogenic_shock_diagnosis = 1

        trop_l = [hadm, gender, initial_I, peak_I, initial_T, peak_T]
        trop_ls.append(trop_l)

        creat_l = [hadm, gender, initial_c]
        creat_ls.append(creat_l)

        shock_l = [hadm, gender, shock_diagnosis]
        shock_ls.append(shock_l)

        cardiogenic_shock_l = [hadm, gender, cardiogenic_shock_diagnosis]
        cardiogenic_shock_ls.append(cardiogenic_shock_l)

    # write ls to df
    trop_headers = ["hadm_id", "gender", "initial_trop_I", "peak_trop_I", "initial_trop_T", "peak_trop_T"]
    trop_df = pd.DataFrame(trop_ls, columns=trop_headers)
    print("trop_df.head() = ", trop_df.head())
    trop_df.to_csv("./data/processed/simplified-ami-patients-troponin-records.csv", index=False)

    creat_headers = ["hadm_id", "gender", "initial_c"]
    creat_df = pd.DataFrame(creat_ls, columns=creat_headers)
    print("creat_df.head() = ", creat_df.head())
    creat_df.to_csv("./data/processed/simplified-ami-patients-creatinine-records.csv", index=False)

    shock_headers = ["hadm_id", "gender", "shock_diag"]
    shocks_df = pd.DataFrame(shock_ls, columns=shock_headers)
    print("shocks_df.head() = ", shocks_df.head())
    shocks_df.to_csv("./data/processed/simplified-ami-patients-shock-diagnoses-records.csv", index=False)

    cardiogenic_shock_headers = ["hadm_id", "gender", "c_shock_diag"]
    c_shocks_df = pd.DataFrame(cardiogenic_shock_ls, columns=cardiogenic_shock_headers)
    print("c_shocks_df.head() = ", c_shocks_df.head())
    c_shocks_df.to_csv("./data/processed/simplified-ami-patients-cardiogenic-shock-diagnoses-records.csv", index=False)


def add_outcome_features():
    """"
    """
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")

    # # get column names of the master file
    # cols = df.columns.tolist()
    # print("cols = ", cols)

    # add discharge locations
    cohort_adm_and_patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
    cohort_adm_and_patient_info_df.columns = [x.lower() for x in cohort_adm_and_patient_info_df.columns]
    discharge_locations_df = cohort_adm_and_patient_info_df[["hadm_id", "discharge_location"]]
    discharge_locations_df["discharge_location"] = discharge_locations_df["discharge_location"].str.lower()
    print("discharge_locations_df head = ", discharge_locations_df.head())

    master_df = pd.merge(df, discharge_locations_df, on="hadm_id", how="left")
    print("master_df's head 1 = ", master_df.head(), sorted(master_df.columns))

    # add troponin values
    troponin_readings_df = pd.read_csv("./data/processed/simplified-ami-patients-troponin-records.csv")
    troponin_readings_df = troponin_readings_df.drop(columns=["gender"])
    print("troponin readings head = ", troponin_readings_df.head())
    master_df = pd.merge(master_df, troponin_readings_df, on="hadm_id", how="left")
    print("master_df's head 2 = ", master_df.head(), sorted(master_df.columns))

    # add creatinine values
    creatinine_readings_df = pd.read_csv("./data/processed/simplified-ami-patients-creatinine-records.csv")
    creatinine_readings_df = creatinine_readings_df.drop(columns=["gender"])
    print("creatinine readings head = ", creatinine_readings_df.head())
    master_df = pd.merge(master_df, creatinine_readings_df, on="hadm_id", how="left")
    print("master_df's head 3 = ", master_df.head(), sorted(master_df.columns))

    # create a feature marking the patient's race as either white or non-white
    master_df["white?"] = "unknown"
    master_df.loc[master_df.ethnicity.isin(["white"]), "white?"] = "yes"
    master_df.loc[master_df.ethnicity.isin(["asian",
                                            "black/african american",
                                            "black/cape verdean",
                                            "hispanic or latino",
                                            "middle eastern",
                                            "multi race ethnicity",
                                            "other", "white - brazilian"]), "white?"] = "no"
    print("master_df's head 6 = ", master_df.head(), sorted(master_df.columns))

    # mark the icd9_code column type as categorical
    master_df = master_df.astype({'icd9_code': 'category'})

    print("master_df dtypes = ", master_df["icd9_code"].dtype, master_df.dtypes.tolist())

    # create a feature on discharge_to_home
    master_df["discharge-to-home?"] = 0
    master_df.loc[master_df.discharge_location == "home", "discharge-to-home?"] = 1

    # add gender/race intersection features
    master_df["white?_yes_f"] = 0
    master_df["white?_yes_m"] = 0
    master_df["white?_unknown_f"] = 0
    master_df["white?_unknown_m"] = 0
    master_df["white?_no_f"] = 0
    master_df["white?_no_m"] = 0

    master_df.loc[(master_df["white?"] == "yes") & (master_df["gender"] == "f"), "white?_yes_f"] = 1
    master_df.loc[(master_df["white?"] == "yes") & (master_df["gender"] == "m"), "white?_yes_m"] = 1
    master_df.loc[(master_df["white?"] == "unknown") & (master_df["gender"] == "f"), "white?_unknown_f"] = 1
    master_df.loc[(master_df["white?"] == "unknown") & (master_df["gender"] == "m"), "white?_unknown_m"] = 1
    master_df.loc[(master_df["white?"] == "no") & (master_df["gender"] == "f"), "white?_no_f"] = 1
    master_df.loc[(master_df["white?"] == "no") & (master_df["gender"] == "m"), "white?_no_m"] = 1

    # group ICD9code features
    master_df["anterior_lateral_mi?"] = 0
    master_df.loc[(master_df["icd9_code"] == 41001) |
                  (master_df["icd9_code"] == 41011) |
                  (master_df["icd9_code"] == 41051), "anterior_lateral_mi?"] = 1

    print("top 5 rows", master_df["icd9_code"].head(), master_df["anterior_lateral_mi?"].head())

    master_df["inferior_posterior_mi?"] = 0
    master_df.loc[(master_df["icd9_code"] == 41021) |
                  (master_df["icd9_code"] == 41031) |
                  (master_df["icd9_code"] == 41041) |
                  (master_df["icd9_code"] == 41042) |
                  (master_df["icd9_code"] == 41061), "inferior_posterior_mi?"] = 1

    master_df["other_mi?"] = 0
    master_df.loc[(master_df["icd9_code"] == 41071) |
                  (master_df["icd9_code"] == 41081) |
                  (master_df["icd9_code"] == 41091), "other_mi?"] = 1

    # create a white?_edited column to make the white?unknown value white?known-not.
    # This is a hack, to make sure that after the dummification of the column, the unknown column is what gets dropped.
    # This allows me to include race information in the model as white?_yes and white?_no

    master_df["white?_edited"] = master_df["white?"]
    master_df.loc[(master_df["white?"] == "unknown"), "white?_edited"] = "known-not"

    print("master_df.head() = ", master_df.head(), len(master_df), master_df.columns.tolist())

    master_df.to_csv("./data/feature-files/ami_patients_features_master.csv", index=False)


# all_patients_df = read_csv_file("./data/raw/healthy_no_pmh_mi_patients_list.csv")
# female_totals, male_totals = male_and_female_population_ami(all_patients_df)
# print("female totals = ", female_totals, " and male totals = ", male_totals)

# organize_patient_procedures()
# write_prescriptions_files("./data/annotated/unique-medications-list-ami-patients_SPC.xls")
# create_patient_features_file()
# clean_troponin_file()
# process_disease_severity_markers()
# add_outcome_features()

