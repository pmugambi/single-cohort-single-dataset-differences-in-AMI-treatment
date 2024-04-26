from __future__ import division
import pandas as pd


def organize_patient_procedures():
    """
    Given a file of all procedures performed on the cohort under study, this function collapses the data so that each
    row represents a unique admission's procedures. As one patient can receive multiple procedures, this function
    enables future analyses of differences per admission by getting all procedures associated with each admission id
    :return: nothing. a consolidated file is written to ./data/processed/
    """
    procedures = pd.read_csv("./data/raw/procedures_on_healthy_ami_patients.csv")
    weights = pd.read_csv("./data/raw/weight_records_ami_patients.csv")
    hadm_ids = procedures['HADM_ID'].unique()

    headers = ["HADM_ID", "age", "gender", "icd9_code", "seq_num", "short_title", "long_title"]
    final_df = pd.DataFrame()
    for hadm_id in hadm_ids:
        # get records for that hadm_id
        hadm_procedures = procedures[procedures["HADM_ID"] == hadm_id][headers].sort_values(
            by="seq_num", axis=0)
        hadm_weights = weights[weights["HADM_ID"] == hadm_id][["VALUE", "VALUEUOM", "CHARTTIME", "label"]].sort_values(
            by="CHARTTIME", axis=0)
        gender = hadm_procedures['gender'].tolist()[0]
        age = hadm_procedures['age'].tolist()[0]

        weight_list = hadm_weights['VALUE'].dropna().tolist()

        # assigning a null value for patients without a recorded weight
        if len(weight_list) > 0:
            weight = round(weight_list[0], 2)
        else:
            weight = None

        # In future, it may be worth checking unit of measurement (UOM).
        # For now, all weights are in kgs so I'll ignore this step
        weightuom_list = hadm_weights['VALUEUOM'].tolist()

        procedures_icd = hadm_procedures['icd9_code'].tolist()
        procedures_short_title = hadm_procedures['short_title'].tolist()
        procedures_long_title = hadm_procedures['long_title'].tolist()

        num_of_procedures = len(hadm_procedures)

        df = pd.DataFrame([[hadm_id, gender, age, weight, num_of_procedures, procedures_icd,
                            procedures_short_title,
                            procedures_long_title]],
                          columns=["hadm_id", "gender", "age", "weight(kg)", "p_count", "icd9_codes", "short_titles",
                                   "long_titles"])
        final_df = pd.concat([final_df, df])
    final_df.to_csv("./data/processed/organized_procedures_for_ami_patients.csv", index=False)


def create_patient_features_file():
    """
    This function takes several raw files (read directly from the database using SQL queries) and merges them to create
    one patient's master file. In this master file all admission, diagnosis, and related comorbidities/disease severity
    markers are merged for easy access and analysis
    :return: nothing. the features are written into a patients'-file and saved in ./data/feature-files/
    """
    info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")[["SUBJECT_ID", "HADM_ID",
                                                                       "INSURANCE", "ETHNICITY",
                                                                       "gender", "age", "admit_duration",
                                                                       "HOSPITAL_EXPIRE_FLAG"]]
    column_names = ["subject_id", "hadm_id", "insurance", "ethnicity", "gender", "age", "admit_duration(days)",
                    "died?"]
    info_df.columns = column_names

    info_df["insurance"] = info_df["insurance"].str.lower()
    info_df["ethnicity"] = info_df["ethnicity"].str.lower()
    info_df["gender"] = info_df["gender"].str.lower()

    diagnoses_df = pd.read_csv("./data/raw/ami-patients-primary-diagnoses.csv")[["HADM_ID", "ICD9_CODE"]]
    diagnoses_df.columns = ["hadm_id", "icd9_code"]

    df = pd.merge(info_df, diagnoses_df, how="outer", on="hadm_id")

    df["non_stemi?"] = 0
    df.loc[df.icd9_code == 41071, "non_stemi?"] = 1

    shock_df = pd.read_csv("./data/raw/ami-patients-shock-diagnoses-records.csv")[["HADM_ID", "ICD9_CODE",
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
    ids_with_pca = pd.read_csv("./data/raw/ami-patients-pca-records.csv")["HADM_ID"].unique().tolist()

    df.loc[df.hadm_id.isin(ids_with_pca), "pca?"] = 1

    ids_with_aspirin_prescriptions = pd.read_csv("./data/raw/ami-patients-aspirin-prescriptions.csv")[
        "HADM_ID"].unique().tolist()
    df["aspirin?"] = 0
    df.loc[df.hadm_id.isin(ids_with_aspirin_prescriptions), "aspirin?"] = 1

    ids_with_liver_conditions = pd.read_csv("./data/raw/ami-patients-with-liver-related-diagnoses.csv")[
        "HADM_ID"].unique().tolist()
    df["liver_conditions?"] = 0
    df.loc[df.hadm_id.isin(ids_with_liver_conditions), "liver_conditions?"] = 1

    ids_with_kidney_conditions = pd.read_csv("./data/raw/ami-patients-with-kidney-related-diagnoses.csv")[
        "HADM_ID"].unique().tolist()
    df["kidney_conditions?"] = 0
    df.loc[df.hadm_id.isin(ids_with_kidney_conditions), "kidney_conditions?"] = 1

    # ids received pain medication
    all_pain_meds, narcotics, _, _, _, _ = assign_medication_types(
        "./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    prescriptions_df = pd.read_csv("./data/raw/ami-patients-prescriptions-all.csv")
    ids_with_pain_med_prescriptions = prescriptions_df[
        prescriptions_df["DRUG"].isin(all_pain_meds)]["HADM_ID"].unique().tolist()

    df["received_pain_med?"] = 0
    df.loc[df.hadm_id.isin(ids_with_pain_med_prescriptions), "received_pain_med?"] = 1

    # # ids received narcotics
    # save narcotics prescriptions file
    narcotics_df = prescriptions_df[prescriptions_df["DRUG"].isin(narcotics)]
    narcotics_df.to_csv("./data/processed/ami-patients-narcotic-prescriptions.csv", index=False)

    ids_with_narcotic_prescriptions = prescriptions_df[
        prescriptions_df["DRUG"].isin(narcotics)]["HADM_ID"].unique().tolist()
    df["received_narcotic?"] = 0
    df.loc[df.hadm_id.isin(ids_with_narcotic_prescriptions), "received_narcotic?"] = 1

    # ids received procedures? look at stephanie's list to see which procedures to eliminate from procedures list
    procedures_df = pd.read_csv("./data/raw/ami-patients-procedures-all.csv")
    surgical, non_surgical, other = assign_procedure_type("./data/annotated/unique-procedures-list-ami-patients.xls")
    surgical_procedures_df = procedures_df[procedures_df["long_title"].isin(surgical)]
    non_surgical_procedures_df = procedures_df[procedures_df["long_title"].isin(non_surgical)]
    other_procedures_df = procedures_df[procedures_df["long_title"].isin(other)]

    non_other_procedures_df = pd.concat([surgical_procedures_df, non_surgical_procedures_df])
    ids_received_procedures = non_other_procedures_df["hadm_id"].unique().tolist()

    df["received_procedure?"] = 0
    df.loc[df.hadm_id.isin(ids_received_procedures), "received_procedure?"] = 1

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

    df.to_csv("./data/feature-files/ami_patients_features_master.csv")


def assign_procedure_type(procedures_file):
    """
    Helper function to assign each medical procedure to one or three categories: surgical, non-surgical, or other.
    Domain experts helped to annotate each unique procedure in the database that was performed on the cohort
    under study. The annotated file is available under ./data/annotated/
    :param procedures_file: annotated file of all procedures performed on the study cohort
    :return: returns procedures in each of the 3 categories
    """
    procedures_df = pd.read_excel(procedures_file, header=None, usecols=[0, 1])
    surgical_procedures = procedures_df[procedures_df[1] == "Surgical "][0].tolist()
    non_surgical_procedures = procedures_df[procedures_df[1] == "Non-surgical"][0].tolist()
    other_procedures = procedures_df[procedures_df[1] == "Other"][0].tolist()
    return surgical_procedures, non_surgical_procedures, other_procedures


def assign_medication_types(medications_file):
    """
    This function is used in the function below (write_prescriptions_files) to mark patients who received
    pain medication, narcotic, etc..
    It takes as input an excel file containing names of unique drugs prescribed to the study cohort, then creates
    drug-type groups and returns them.
    :param medications_file: an excel file containing names of unique drugs prescribed to the study cohort, annotated
    by a domain expert. the annotated file is available in ./data/annotated/
    :return: drug-type groups
    """
    #
    medications_df = pd.read_excel(medications_file, header=None)
    pain_medications_df = medications_df[medications_df[3].isin(["YES", "MAYBE"])]
    pain_meds = pain_medications_df[0].unique().tolist()
    narcotics = pain_medications_df[pain_medications_df[4] == "YES"][0].unique().tolist()
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
    beta_blockers_df.to_csv("./data/processed/ami-patients-beta-blockers-prescriptions.csv", index=False)
    statin_df.to_csv("./data/processed/ami-patients-statins-prescriptions.csv", index=False)
    anti_platelets_df.to_csv("./data/processed/ami-patients-anti-platelets-prescriptions.csv", index=False)
    ace_inhibitors_df.to_csv("./data/processed/ami-patients-ace-inhibitors-prescriptions.csv", index=False)


def clean_troponin_file():
    """
    Cleans the raw troponin records file (read directly from the EHR database using SQL) to mark values greater-than or
    less-than as the ceiling/floor value. That is, the string values are converted into a quantity to enable
    further quantitative analysis
    :return: nothing. the clean file is written to ./data/processed/
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
    values = [float(x) for x in values]

    df["values_mod"] = values
    df.to_csv("./data/processed/ami-patients-troponin-records-modified.csv", index=False)


def process_disease_severity_markers():
    """
    Function processes troponin, creatinine, and shock data for each admission id. It helps to obtain the initial,
    and peak troponin values for each admission id (i.e., read all recorded troponin values and obtain the very
    first, i.e., initial, and the max, i.e., peak). Additionally it obtain initial creatinine recorded value,
    and determines if the patient experienced shock (and separately cardiogenic shock) at any point of their admission.
    :return: nothing. all generated values as written to files in ./data/processed/
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

    # group records by hadm_id
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

        gender = patients_df[patients_df["hadm_id"] == hadm]["gender"].tolist()[0]
        gender = gender.lower()
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

        if len(hadm_troponin_T_df) > 0:
            # sort df by date, i.e. charttime, asc
            sorted_hadm_troponin_T_df = hadm_troponin_T_df.sort_values(by="charttime",
                                                                       ascending=True)
            initial_T = sorted_hadm_troponin_T_df.head(1)["values_mod"].tolist()[0]
            peak_T = max(sorted_hadm_troponin_T_df["values_mod"])

        # process creatinine averages here
        if len(hadm_creatinine_only_df) > 0:
            # sort df by date, i.e. charttime, asc
            sorted_hadm_creatinine_only_df = hadm_creatinine_only_df.sort_values(by="charttime", ascending=True)
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
    trop_df.to_csv("./data/processed/simplified-ami-patients-troponin-records.csv", index=False)

    creat_headers = ["hadm_id", "gender", "initial_c"]
    creat_df = pd.DataFrame(creat_ls, columns=creat_headers)
    creat_df.to_csv("./data/processed/simplified-ami-patients-creatinine-records.csv", index=False)

    shock_headers = ["hadm_id", "gender", "shock_diag"]
    shocks_df = pd.DataFrame(shock_ls, columns=shock_headers)
    shocks_df.to_csv("./data/processed/simplified-ami-patients-shock-diagnoses-records.csv", index=False)

    cardiogenic_shock_headers = ["hadm_id", "gender", "c_shock_diag"]
    c_shocks_df = pd.DataFrame(cardiogenic_shock_ls, columns=cardiogenic_shock_headers)
    c_shocks_df.to_csv("./data/processed/simplified-ami-patients-cardiogenic-shock-diagnoses-records.csv", index=False)


def add_outcome_features():
    """
    This function add the troponin, creatinine values to the master patients' file. Additionally, ethnicity, type of MI,
    and discharge location values are re-categorized into fewer groups for when models are fit using this feature file.
    :return: nothing. new features are written to the patients master file
    """
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")

    # add discharge locations
    cohort_adm_and_patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
    cohort_adm_and_patient_info_df.columns = [x.lower() for x in cohort_adm_and_patient_info_df.columns]
    discharge_locations_df = cohort_adm_and_patient_info_df[["hadm_id", "discharge_location"]]
    discharge_locations_df["discharge_location"] = discharge_locations_df["discharge_location"].str.lower()

    master_df = pd.merge(df, discharge_locations_df, on="hadm_id", how="left")

    # add troponin values
    troponin_readings_df = pd.read_csv("./data/processed/simplified-ami-patients-troponin-records.csv")
    troponin_readings_df = troponin_readings_df.drop(columns=["gender"])
    master_df = pd.merge(master_df, troponin_readings_df, on="hadm_id", how="left")

    # add creatinine values
    creatinine_readings_df = pd.read_csv("./data/processed/simplified-ami-patients-creatinine-records.csv")
    creatinine_readings_df = creatinine_readings_df.drop(columns=["gender"])
    master_df = pd.merge(master_df, creatinine_readings_df, on="hadm_id", how="left")

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

    # mark the icd9_code column type as categorical
    master_df = master_df.astype({'icd9_code': 'category'})

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

    master_df.to_csv("./data/feature-files/ami_patients_features_master.csv", index=False)


if __name__ == '__main__':
    organize_patient_procedures()
    write_prescriptions_files("./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    create_patient_features_file()
    clean_troponin_file()
    process_disease_severity_markers()
    add_outcome_features()

