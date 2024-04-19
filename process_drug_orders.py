import pandas as pd
from datetime import timedelta
from data_processing import assign_medication_types
from utils import pad_list_with_zeros


def compute_dosage_per_day(dosage_df):
    """
    Distributes the prescription amount (in dosage days) to all the days in the prescription
    :param dosage_df: The DF of all patients' prescriptions with each prescription amount quantified in dosage days
    :return: A DF of all patients' prescriptions with each day in each prescription assign a 'dosage_days' value
    """
    df = pd.DataFrame()
    dosage_df = dosage_df[pd.notnull(dosage_df["startdate"])]
    dosage_df = dosage_df[pd.notnull(dosage_df["enddate"])]
    for i in range(0, len(dosage_df)):
        row = dosage_df.iloc[[i]]
        startdate = row["startdate"]
        enddate = row["enddate"]
        startdate = pd.to_datetime(startdate, format="%Y-%m-%d %H:%M:%S")
        enddate = pd.to_datetime(enddate, format="%Y-%m-%d %H:%M:%S")
        delta = int((enddate - startdate).astype('timedelta64[D]'))

        for j in range(delta + 1):
            day = startdate + timedelta(days=j)
            row["date"] = day
            df = pd.concat([df, row])
    return df


def sum_mme_per_day(days_mme_df, write_filename):  # for each day when a patient has a prescription
    """
    Given a DF containing all patients' prescriptions and their dosage days, sum dosage days for each date.
    :param days_mme_df: A DF containing all patients' prescriptions with their associated dosage days
    :return: A DF with all patients' prescription dosage days for each date with a prescription
    """
    df = pd.DataFrame()
    hadm_ids = list(days_mme_df["hadm_id"].unique())
    for hadm_id in hadm_ids:
        group = days_mme_df.where(days_mme_df["hadm_id"] == hadm_id)
        group = group[pd.notnull(group['hadm_id'])]

        group_dates = list(group["date"].unique())
        for date in group_dates:
            date_records = group.where(group["date"] == date)
            date_records = date_records[pd.notnull(date_records["hadm_id"])]

            total_mme = sum(date_records["mme_value"])

            # narcotics_records = date_records.where(date_records["is_narcotic"] == "yes")
            # narcotics_records = narcotics_records[pd.notnull(narcotics_records["HADM_ID"])]
            # total_narcotic_dosage_days = sum(narcotics_records["dosage_days"])

            # non_narcotics_records = date_records.where(date_records["is_narcotic"] == "no")
            # non_narcotics_records = non_narcotics_records[pd.notnull(non_narcotics_records["HADM_ID"])]
            # total_non_narcotic_dosage_days = sum(non_narcotics_records["dosage_days"])

            date_record_details = date_records.head(1)

            date_record_details = date_record_details.drop("drug", axis=1)
            date_record_details = date_record_details.drop("startdate", axis=1)
            date_record_details = date_record_details.drop("enddate", axis=1)
            date_record_details = date_record_details.drop("dose_val_rx", axis=1)
            date_record_details = date_record_details.drop("dose_unit_rx", axis=1)
            date_record_details = date_record_details.drop("route", axis=1)
            date_record_details = date_record_details.drop("drug_type", axis=1)
            date_record_details = date_record_details.drop("drug_name_poe", axis=1)
            date_record_details = date_record_details.drop("drug_name_generic", axis=1)
            date_record_details = date_record_details.drop("formulary_drug_cd", axis=1)
            date_record_details = date_record_details.drop("gsn", axis=1)
            date_record_details = date_record_details.drop("ndc", axis=1)
            date_record_details = date_record_details.drop("prod_strength", axis=1)
            date_record_details = date_record_details.drop("form_val_disp", axis=1)
            date_record_details = date_record_details.drop("form_unit_disp", axis=1)

            date_record_details["total_mme"] = total_mme
            df = pd.concat([df, date_record_details])
    df.to_csv(write_filename)
    return df


def assign_mme_values(df):
    """
    Assigns dosage days to each prescription in the patients' prescription <df>
    :param df: All patients' prescriptions
    :return: All patients' prescriptions with dosage days attached to each prescription
    """
    data = df[pd.notnull(df['dose_val_rx'])]
    dose_amounts = list(data["dose_val_rx"])
    drugs = list(data["drug"])
    dose_units = list(data["dose_unit_rx"])
    # genders = list(data["gender"])
    routes = list(data["route"])
    product_strengths = list(data["prod_strength"])

    mme_list = []
    # gender_numbers = []

    for i in range(0, len(drugs)):
        drug = drugs[i].lower()
        dose_amount = str(dose_amounts[i])
        dose_unit = dose_units[i].lower()
        # gender = genders[i].lower()
        route = routes[i].lower()
        if '-' in dose_amount:
            dose_amount = dose_amount.split("-")[0]
        if ',' in dose_amount:
            dose_amount = dose_amount.replace(",", "")
        dose_amount = float(dose_amount)
        product_strength = product_strengths[i]

        if drug == "acetaminophen w/codeine":
            # TODO: I looked at the list of patients with oxycodeine and they all had 30mg of oxycodeine.
            #  This needs to be made dynamic instead of hard coding
            codeine_mg = product_strength.split("/")[1]
            print("codeine_mg = ", codeine_mg)
            mme = 15 * 3  # the document shows that 10mg of oxycodeine is 15mg PO morphine equivalent.
        elif drug == "fentanyl citrate" and dose_unit == "mcg" and route in ["iv", "iv drip"]:
            if dose_amount == 12.5:
                mme = 3.75
            elif dose_amount == 25:
                mme = 7.5
            elif dose_amount == 50:
                mme = 15
            elif dose_amount == 100:
                mme = 30
            else:
                mme = (dose_amount / 100) * 30
        elif drug == "fentanyl citrate" and dose_unit == "mg" and route in ["iv", "iv drip"]:
            dose_amount_mcg = dose_amount * 1000
            mme = (dose_amount_mcg / 100) * 30
        elif drug == "fentanyl patch":
            mme = dose_amount * 7.2  # conversion factor read from
            # https://www.cms.gov/Medicare/Prescription-Drug-coverage/PrescriptionDrugCovContra/Downloads/
            # Opioid-Morphine-EQ-Conversion-Factors-Aug-2017.pdf

        # elif drug == "fentanyl citrate" and dose_unit == "mcg" and route == "pb":
        #     days = dose_amount / float(2500)

        elif drug == "hydrocodone-acetaminophen":
            # TODO: I hard coded the MME value here too. For all patients, the hydrocodone content in this drug was
            #  5mg, which converts to 5 mme according to the MME doc from the MD.
            mme = 5
        elif "hydromorphone" in drug and dose_unit == "mg" and route in ["po", "po/ng"]:
            mme = (dose_amount / 4) * 16  # the doc says that 4mg PO has a MME of 16. Hence the multiplier
        elif "hydromorphone" in drug and dose_unit == "mg" and route == "sc":
            sc_to_po_dose = dose_amount * 5  # The doc shows that IM to PO for hydrophone is 1.5mg to 7.5mg
            # which is a multiplier factor of 5, hence the multiplier
            # Once the dose is in PO, we convert to MME as with step above
            mme = (sc_to_po_dose / 4) * 16
        elif "hydromorphone" in drug and dose_unit == "mg" and route == "iv":
            mme = dose_amount * 20  # the document days that 1mg IV has a MME of 20. Hence the multiplier
        elif "hydromorphone" in drug and route == "ivpca":
            # days = dose_amount / float(1)
            mme = (dose_amount / 30) * 600  # the doc says that 30mh PCA hydromorphone is equivalent to 600MME.
            # Hence the division by 30 and multiply by 600
        elif "meperidine" in drug and route == "iv":
            meperidine_po_dose = dose_amount * 4  # the doc says that 75mg of IV/IM meperidine is equivalent to 300mg PO
            # meperidine, which is a multiplier factor of 4.
            # Then, to convert to MME, the link under fantanyl patch says that its a 0.1 conversion factor
            mme = meperidine_po_dose * 0.1
        elif "morphine sulfate" in drug and dose_unit == "mg" and route is not "po":
            mme = dose_amount * 3  # the doc says that 10mg of morphine IV/IM are equivalent to 30 mg of morphine PO,
            # hence a multiplier factor of 3
        elif "morphine sulfate" in drug and dose_unit == "mg" and route == "po":
            mme = dose_amount

        elif drug in ["oxycodone", "oxycodone sr (oxycontin)", "oxycodone (immediate release) "] \
                and route in ["po", "po/ng"]:
            mme = (dose_amount / 5) * 7.5  # using the doc 5mg of ocycodone PO has a MME of 7.5, hence the divide
            # then multiplication.
            # TODO: there is a confusion in the doc about the IM/IV conversion and the drug to MME conversion.
            #  Check with the MD
        elif "oxycodone-acetaminophen" in drug:
            mme = 7.5  # the doc says that 5mg oxy PO has a MME of 7.5. All the patients had a 5mg/325 tab i.e.
            # 5mg of oxy and 325 of acetaminophen. Hence the 7.5
            # TODO: I'm hardcoding the MME value here too, because all patients were prescribed the 5mg/325mg tablet.
            #  Future analyses may require changing this to dynamic by reading the product strength
        elif "tramadol" in drug:
            mme = (dose_amount / 50) * 5  # the doc says that 50mg tramadol has a MME of 5.
            # Hence the division and multiplication
        elif "propoxyphene" in drug:
            mme = dose_amount * 100 * 0.23  # the paper shared by the MD says the conversion factor is 0.23.
            # Searching the internet showed that each tablet had a 100mg propoxyphene amount. Hence the multiplication.
            # For this particular drug, the dosage unit is the number of tablets, rather than mgs.
        elif "methadon" in drug and route == "po" and dose_unit == "mg":
            # NOTE: all AMI patients that were prescribed methadone, it was administered via PO.
            # Still, I checked to the route and dose_unti
            mme = dose_amount * 1.5
        else:
            mme = -1

        mme_list.append(mme)

        # if gender == "m":
        #     gender_num = 0
        # else:
        #     gender_num = 1
        # gender_numbers.append(gender_num)
    data["mme_value"] = mme_list
    # data["gender_num"] = gender_numbers
    return data


def mme_per_admit_day(patient_info_df, patients_total_daily_mme_df, column_name):  # for each admit day of the patient,
    # whether they had a prescription or not
    """
    Given a DF of all patients' prescriptions dosage days summed per date, computes the dosage days per admit day.
    That is, it looks through the entire patient's admit duration and for each day either assigns a 0
    (if no records for that date were found), or reads the values from the total daily dosage DF.
    :param column_name: The column name of the column containing total_dosage_days
    :param patients_total_daily_mme_df: A list of all patients' dosage days aggregated per date i.e. a List of lists,
     where each list in the List is a single patient's dosage days per admit day
    :return: A DF of all patients' dosage days for each admit day
    """
    all_patient_mme_days = []
    hadm_ids = patients_total_daily_mme_df["hadm_id"].unique().tolist()
    for hadm_id in hadm_ids:
        mme_days = []
        group_info = patient_info_df.where(patient_info_df["HADM_ID"] == hadm_id)
        group_info = group_info[pd.notnull(group_info['HADM_ID'])]

        admittime = group_info["ADMITTIME"].tolist()[0]
        dischargetime = group_info["DISCHTIME"].tolist()[0]

        group = patients_total_daily_mme_df.where(patients_total_daily_mme_df["hadm_id"] == hadm_id)
        group = group[pd.notnull(group['hadm_id'])]

        # print("admittime = ", admittime, " dischargetime = ", dischargetime)
        # print("group = ", group)

        admittime = pd.to_datetime(admittime, format="%Y-%m-%d %H:%M:%S")
        dischargetime = pd.to_datetime(dischargetime, format="%Y-%m-%d %H:%M:%S")
        admit_duration = (dischargetime - admittime).days

        for j in range(admit_duration + 1):
            day = admittime.date() + timedelta(days=j)
            day = pd.to_datetime([day])[0]
            group["date"] = pd.to_datetime(group["date"])
            day_records = group.where(group["date"] == day)
            day_records = day_records[pd.notnull(day_records['hadm_id'])]
            # print("day = ", day, " day records = ", day_records)

            if len(day_records) == 0:
                total_mme_days = 0
            else:
                total_mme_days = day_records[column_name].tolist()[0]

            # print("group = ", hadm_id, " day = ", day, "total_dosage = ", total_dosage_days)
            mme_days.append(total_mme_days)
        all_patient_mme_days.append(mme_days)
    return hadm_ids, all_patient_mme_days


def process_narcotic_prescriptions():
    """

    :return:
    """
    df = pd.read_csv("./data/processed/ami-patients-narcotic-prescriptions.csv")
    df.columns = [x.lower() for x in df.columns]
    print("df cols = ", df.columns)

    df_with_mme = assign_mme_values(df)

    df_with_mme = df_with_mme[df_with_mme.columns.drop(list(df_with_mme.filter(regex='unnamed')))]
    print("df with mme head = ", df_with_mme.head(), len(df_with_mme))

    df_with_mme.to_csv("./data/processed/narcotic-prescriptions-with-mme-values.csv")

    df_with_mme_per_day = compute_dosage_per_day(dosage_df=df_with_mme)
    print("expanded df head = ", df_with_mme_per_day.head(), len(df_with_mme_per_day))

    df_with_mme_per_day.to_csv("./data/processed/expanded-narcotics-prescriptions-with-mme-values.csv")

    summed_patient_mme_df = sum_mme_per_day(df_with_mme_per_day, "./data/processed/per-day-summed-mme-values.csv")
    print("summed_patient_mme_df = ", summed_patient_mme_df.head(), len(summed_patient_mme_df))

    # mme_per_admit_day
    patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
    hadms, mme_doses = mme_per_admit_day(patient_info_df=patient_info_df,
                                         patients_total_daily_mme_df=summed_patient_mme_df,
                                         column_name="total_mme")

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
    merged_df.to_csv("./data/processed/per-patient-per-day-mme-values.csv")


def sum_non_narcs_per_day(days_mme_df, write_filename):  # for each day when a patient has a prescription
    """
    Given a DF containing all patients' prescriptions and their dosage days, sum dosage days for each date.
    :param days_mme_df: A DF containing all patients' prescriptions with their associated dosage days
    :return: A DF with all patients' prescription dosage days for each date with a prescription
    """
    df = pd.DataFrame()
    hadm_ids = list(days_mme_df["hadm_id"].unique())
    for hadm_id in hadm_ids:
        group = days_mme_df.where(days_mme_df["hadm_id"] == hadm_id)
        group = group[pd.notnull(group['hadm_id'])]
        group_dates = list(group["date"].unique())
        for date in group_dates:
            date_records = group.where(group["date"] == date)
            date_records = date_records[pd.notnull(date_records["hadm_id"])]

            total_orders = len(date_records)

            date_record_details = date_records.head(1)

            date_record_details = date_record_details.drop("drug", axis=1)
            date_record_details = date_record_details.drop("startdate", axis=1)
            date_record_details = date_record_details.drop("enddate", axis=1)
            date_record_details = date_record_details.drop("dose_val_rx", axis=1)
            date_record_details = date_record_details.drop("dose_unit_rx", axis=1)
            date_record_details = date_record_details.drop("route", axis=1)
            date_record_details = date_record_details.drop("drug_type", axis=1)
            date_record_details = date_record_details.drop("drug_name_poe", axis=1)
            date_record_details = date_record_details.drop("drug_name_generic", axis=1)
            date_record_details = date_record_details.drop("formulary_drug_cd", axis=1)
            date_record_details = date_record_details.drop("gsn", axis=1)
            date_record_details = date_record_details.drop("ndc", axis=1)
            date_record_details = date_record_details.drop("prod_strength", axis=1)
            date_record_details = date_record_details.drop("form_val_disp", axis=1)
            date_record_details = date_record_details.drop("form_unit_disp", axis=1)

            date_record_details["total_orders"] = total_orders
            df = pd.concat([df, date_record_details])
    df.to_csv(write_filename)
    return df


def process_non_narcotic_prescriptions():
    pain_meds, narcs, _, _, _, _ = assign_medication_types(
        "./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    non_narcs = set(pain_meds) - set(narcs)

    prescriptions_df = pd.read_csv("./data/raw/ami-patients-prescriptions-all.csv")
    non_narc_df = prescriptions_df[prescriptions_df["DRUG"].isin(non_narcs)]
    non_narc_df.columns = [col.lower() for col in non_narc_df.columns]

    non_narc_df.to_csv("./data/processed/ami-patients-non-narcotic-pain-prescriptions.csv")

    print(non_narc_df.head(), len(non_narc_df))

    nn_df_per_day = compute_dosage_per_day(dosage_df=non_narc_df)
    print("expanded df head = ", nn_df_per_day.head(), len(nn_df_per_day))

    nn_df_per_day.to_csv("./data/processed/expanded-non-narcotic-prescriptions.csv")

    # sum_non_narcs_per_day
    summed_patient_nn_doses_df = sum_non_narcs_per_day(nn_df_per_day,
                                                       "./data/processed/per-day-summed-non-narc-dose-values.csv")
    print("summed_patient_nn_doses_df = ", summed_patient_nn_doses_df.head(), len(summed_patient_nn_doses_df))

    patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
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
    merged_df.to_csv("./data/processed/per-patient-per-day-nn-dose-values.csv")


def sum_aspirin_per_day(days_mme_df, write_filename):  # for each day when a patient has a prescription
    """
    Given a DF containing all patients' prescriptions and their dosage days, sum dosage days for each date.
    :param days_mme_df: A DF containing all patients' prescriptions with their associated dosage days
    :return: A DF with all patients' prescription dosage days for each date with a prescription
    """
    df = pd.DataFrame()
    hadm_ids = list(days_mme_df["hadm_id"].unique())
    for hadm_id in hadm_ids:
        group = days_mme_df.where(days_mme_df["hadm_id"] == hadm_id)
        group = group[pd.notnull(group['hadm_id'])]
        group_dates = list(group["date"].unique())
        for date in group_dates:
            date_records = group.where(group["date"] == date)
            date_records = date_records[pd.notnull(date_records["hadm_id"])]
            total_dose = sum(date_records["dose_val_rx"])

            # TODO: compute total day's aspirin dosage

            date_record_details = date_records.head(1)

            date_record_details = date_record_details.drop("drug", axis=1)
            date_record_details = date_record_details.drop("startdate", axis=1)
            date_record_details = date_record_details.drop("enddate", axis=1)
            date_record_details = date_record_details.drop("dose_val_rx", axis=1)
            date_record_details = date_record_details.drop("dose_unit_rx", axis=1)
            date_record_details = date_record_details.drop("route", axis=1)
            date_record_details = date_record_details.drop("drug_type", axis=1)
            date_record_details = date_record_details.drop("drug_name_poe", axis=1)
            date_record_details = date_record_details.drop("drug_name_generic", axis=1)
            date_record_details = date_record_details.drop("formulary_drug_cd", axis=1)
            date_record_details = date_record_details.drop("gsn", axis=1)
            date_record_details = date_record_details.drop("ndc", axis=1)
            date_record_details = date_record_details.drop("prod_strength", axis=1)
            date_record_details = date_record_details.drop("form_val_disp", axis=1)
            date_record_details = date_record_details.drop("form_unit_disp", axis=1)

            date_record_details["total_dose"] = total_dose
            df = pd.concat([df, date_record_details])
    df.to_csv(write_filename)
    return df


def process_aspirin_prescriptions():
    aspirin_prescriptions_df = pd.read_csv("./data/raw/ami-patients-aspirin-prescriptions.csv")
    aspirin_prescriptions_df.columns = [col.lower() for col in aspirin_prescriptions_df.columns]
    print(aspirin_prescriptions_df.head(), len(aspirin_prescriptions_df))

    aspirin_df_per_day = compute_dosage_per_day(dosage_df=aspirin_prescriptions_df)
    print("expanded df head = ", aspirin_df_per_day.head(), len(aspirin_df_per_day))

    aspirin_df_per_day.to_csv("./data/processed/expanded-aspirin-prescriptions.csv")

    summed_patient_aspirin_orders_df = sum_aspirin_per_day(aspirin_df_per_day,
                                                           "./data/processed/per-day-summed-aspirin-order-values.csv")
    print("summed_patient_aspirin_orders_df = ", summed_patient_aspirin_orders_df.head(),
          len(summed_patient_aspirin_orders_df))

    patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
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
    merged_df.to_csv("./data/processed/per-patient-per-day-aspirin-order-values.csv")


def process_drug_prescriptions(drug_class):
    """

    :param drug_class:
    :return:
    """
    prescriptions_df = None
    if drug_class == "statin":
        prescriptions_df = pd.read_csv("./data/processed/ami-patients-statins-prescriptions.csv")
    elif drug_class == "anti-platelet":
        prescriptions_df = pd.read_csv("./data/processed/ami-patients-anti-platelets-prescriptions.csv")
    elif drug_class == "beta-blocker":
        prescriptions_df = pd.read_csv("./data/processed/ami-patients-beta-blockers-prescriptions.csv")
    elif drug_class == "ace-inhibitor":
        prescriptions_df = pd.read_csv("./data/processed/ami-patients-ace-inhibitors-prescriptions.csv")
    prescriptions_df.columns = [col.lower() for col in prescriptions_df.columns]
    print(prescriptions_df.head(), len(prescriptions_df))

    drug_df_per_day = compute_dosage_per_day(dosage_df=prescriptions_df)
    print("expanded df head = ", drug_df_per_day.head(), len(drug_df_per_day))

    drug_df_per_day.to_csv("./data/processed/expanded-" + drug_class + "-prescriptions.csv")

    summed_patient_drug_orders_df = sum_non_narcs_per_day(drug_df_per_day,
                                                          "./data/processed/per-day-summed-" + drug_class +
                                                          "-order-values.csv")
    print("summed_patient_drug_orders_df = ", summed_patient_drug_orders_df.head(),
          len(summed_patient_drug_orders_df))

    patient_info_df = pd.read_csv("./data/raw/ami-patients-admissions.csv")
    hadms, drug_orders = mme_per_admit_day(patient_info_df=patient_info_df,
                                           patients_total_daily_mme_df=summed_patient_drug_orders_df,
                                           column_name="total_orders")

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
    merged_df.to_csv("./data/processed/per-patient-per-day-" + drug_class + "-order-values.csv")


def drug_per_day_patient_features(df, filename, drug_class):
    """

    :param df:
    :param filename:
    :param drug_class:
    :return:
    """
    df["received_" + drug_class + "?"] = 0
    ids_with_drug_class = pd.read_csv(filename)["hadm_id"].unique().tolist()
    print("ids_with_drug_class_ids = ", ids_with_drug_class)
    df.loc[df.hadm_id.isin(ids_with_drug_class), "received_" + drug_class + "?"] = 1

    drug_orders_df = pd.read_csv(filename)
    drug_doses_df_day_1 = drug_orders_df[drug_orders_df["day1"] > 0]
    drug_doses_df_day_2 = drug_orders_df[drug_orders_df["day2"] > 0]
    drug_doses_df_day_3 = drug_orders_df[drug_orders_df["day3"] > 0]
    drug_doses_df_day_4 = drug_orders_df[drug_orders_df["day4"] > 0]
    drug_doses_df_day_5 = drug_orders_df[drug_orders_df["day5"] > 0]
    ids_with_drug_doses_d1 = drug_doses_df_day_1["hadm_id"].unique()
    d1_drug_doses = drug_doses_df_day_1[["hadm_id", "day1"]]
    d1_drug_doses.columns = ["hadm_id", "d1_" + drug_class[0:2] + "d"]

    ids_with_drug_doses_d2 = drug_doses_df_day_2["hadm_id"].unique()
    d2_drug_doses = drug_doses_df_day_2[["hadm_id", "day2"]]
    d2_drug_doses.columns = ["hadm_id", "d2_" + drug_class[0:2] + "d"]

    ids_with_drug_doses_d3 = drug_doses_df_day_3["hadm_id"].unique()
    d3_drug_doses = drug_doses_df_day_3[["hadm_id", "day3"]]
    d3_drug_doses.columns = ["hadm_id", "d3_" + drug_class[0:2] + "d"]

    ids_with_drug_doses_d4 = drug_doses_df_day_4["hadm_id"].unique()
    d4_drug_doses = drug_doses_df_day_4[["hadm_id", "day4"]]
    d4_drug_doses.columns = ["hadm_id", "d4_" + drug_class[0:2] + "d"]

    ids_with_drug_doses_d5 = drug_doses_df_day_5["hadm_id"].unique()
    d5_drug_doses = drug_doses_df_day_5[["hadm_id", "day5"]]
    d5_drug_doses.columns = ["hadm_id", "d5_" + drug_class[0:2] + "d"]

    total_ad_drug_dose = drug_orders_df[["hadm_id", drug_class + "-orders"]]

    df[drug_class + "_day1?"] = 0
    df.loc[df.hadm_id.isin(ids_with_drug_doses_d1), drug_class + "_day1?"] = 1
    df = pd.merge(df, d1_drug_doses, on="hadm_id", how="left")

    df[drug_class + "_day2?"] = 0
    df.loc[df.hadm_id.isin(ids_with_drug_doses_d2), drug_class + "_day2?"] = 1
    df = pd.merge(df, d2_drug_doses, on="hadm_id", how="left")

    df[drug_class + "_day3?"] = 0
    df.loc[df.hadm_id.isin(ids_with_drug_doses_d3), drug_class + "_day3?"] = 1
    df = pd.merge(df, d3_drug_doses, on="hadm_id", how="left")

    df[drug_class + "_day4?"] = 0
    df.loc[df.hadm_id.isin(ids_with_drug_doses_d4), drug_class + "_day4?"] = 1
    df = pd.merge(df, d4_drug_doses, on="hadm_id", how="left")

    df[drug_class + "_day5?"] = 0
    df.loc[df.hadm_id.isin(ids_with_drug_doses_d5), drug_class + "_day5?"] = 1
    df = pd.merge(df, d5_drug_doses, on="hadm_id", how="left")

    df = pd.merge(df, total_ad_drug_dose, on="hadm_id", how="left")
    # TODO: to simplify, can just merge mme_values with df and set 0 to nan

    print("names = ", df.columns.tolist())

    # average aspirin orders
    df["ad-average-" + drug_class] = round(df[drug_class + "-orders"] / df["admit_duration(days)"], 2)
    print("df.head = ", df.head())
    return df


def add_drug_features_in_patient_file():
    """

    :return:
    """
    df = pd.read_csv("data/feature-files/ami_patients_features_master.csv")
    # ids with opiod use disorder/ substance abuse problem

    # mme values
    mme_values_df = pd.read_csv("./data/processed/per-patient-per-day-mme-values.csv")
    mme_values_df_day_1 = mme_values_df[mme_values_df["day1"] > 0]
    mme_values_df_day_2 = mme_values_df[mme_values_df["day2"] > 0]
    mme_values_df_day_3 = mme_values_df[mme_values_df["day3"] > 0]
    mme_values_df_day_4 = mme_values_df[mme_values_df["day4"] > 0]
    mme_values_df_day_5 = mme_values_df[mme_values_df["day5"] > 0]
    ids_with_mme_values_d1 = mme_values_df_day_1["hadm_id"].unique()
    d1_mme_values = mme_values_df_day_1[["hadm_id", "day1"]]
    d1_mme_values.columns = ["hadm_id", "d1_mme"]

    ids_with_mme_values_d2 = mme_values_df_day_2["hadm_id"].unique()
    d2_mme_values = mme_values_df_day_2[["hadm_id", "day2"]]
    d2_mme_values.columns = ["hadm_id", "d2_mme"]

    ids_with_mme_values_d3 = mme_values_df_day_3["hadm_id"].unique()
    d3_mme_values = mme_values_df_day_3[["hadm_id", "day3"]]
    d3_mme_values.columns = ["hadm_id", "d3_mme"]

    ids_with_mme_values_d4 = mme_values_df_day_4["hadm_id"].unique()
    d4_mme_values = mme_values_df_day_4[["hadm_id", "day4"]]
    d4_mme_values.columns = ["hadm_id", "d4_mme"]

    ids_with_mme_values_d5 = mme_values_df_day_5["hadm_id"].unique()
    d5_mme_values = mme_values_df_day_5[["hadm_id", "day5"]]
    d5_mme_values.columns = ["hadm_id", "d5_mme"]

    total_ad_mme = mme_values_df[["hadm_id", "ad-total-mme"]]

    df["received_narc_day1?"] = 0
    df.loc[df.hadm_id.isin(ids_with_mme_values_d1), "received_narc_day1?"] = 1
    df = pd.merge(df, d1_mme_values, on="hadm_id", how="left")

    df["received_narc_day2?"] = 0
    df.loc[df.hadm_id.isin(ids_with_mme_values_d2), "received_narc_day2?"] = 1
    df = pd.merge(df, d2_mme_values, on="hadm_id", how="left")

    df["received_narc_day3?"] = 0
    df.loc[df.hadm_id.isin(ids_with_mme_values_d3), "received_narc_day3?"] = 1
    df = pd.merge(df, d3_mme_values, on="hadm_id", how="left")

    df["received_narc_day4?"] = 0
    df.loc[df.hadm_id.isin(ids_with_mme_values_d4), "received_narc_day4?"] = 1
    df = pd.merge(df, d4_mme_values, on="hadm_id", how="left")

    df["received_narc_day5?"] = 0
    df.loc[df.hadm_id.isin(ids_with_mme_values_d5), "received_narc_day5?"] = 1
    df = pd.merge(df, d5_mme_values, on="hadm_id", how="left")

    df = pd.merge(df, total_ad_mme, on="hadm_id", how="left")
    # note: to simplify, can just merge mme_values with df and set 0 to nan

    # average mme
    df["ad-average-mme"] = round(df["ad-total-mme"] / df["admit_duration(days)"], 2)

    # non-narcotic dose values
    nn_doses_df = pd.read_csv("./data/processed/per-patient-per-day-nn-dose-values.csv")
    nn_doses_df_day_1 = nn_doses_df[nn_doses_df["day1"] > 0]
    nn_doses_df_day_2 = nn_doses_df[nn_doses_df["day2"] > 0]
    nn_doses_df_day_3 = nn_doses_df[nn_doses_df["day3"] > 0]
    nn_doses_df_day_4 = nn_doses_df[nn_doses_df["day4"] > 0]
    nn_doses_df_day_5 = nn_doses_df[nn_doses_df["day5"] > 0]
    ids_with_nn_doses_d1 = nn_doses_df_day_1["hadm_id"].unique()
    d1_nn_doses = nn_doses_df_day_1[["hadm_id", "day1"]]
    d1_nn_doses.columns = ["hadm_id", "d1_nnd"]

    ids_with_nn_doses_d2 = nn_doses_df_day_2["hadm_id"].unique()
    d2_nn_doses = nn_doses_df_day_2[["hadm_id", "day2"]]
    d2_nn_doses.columns = ["hadm_id", "d2_nnd"]

    ids_with_nn_doses_d3 = nn_doses_df_day_3["hadm_id"].unique()
    d3_nn_doses = nn_doses_df_day_3[["hadm_id", "day3"]]
    d3_nn_doses.columns = ["hadm_id", "d3_nnd"]

    ids_with_nn_doses_d4 = nn_doses_df_day_4["hadm_id"].unique()
    d4_nn_doses = nn_doses_df_day_4[["hadm_id", "day4"]]
    d4_nn_doses.columns = ["hadm_id", "d4_nnd"]

    ids_with_nn_doses_d5 = nn_doses_df_day_5["hadm_id"].unique()
    d5_nn_doses = nn_doses_df_day_5[["hadm_id", "day5"]]
    d5_nn_doses.columns = ["hadm_id", "d5_nnd"]

    total_ad_nn_dose = nn_doses_df[["hadm_id", "ad-total-orders"]]

    df["received_n_narc_day1?"] = 0
    df.loc[df.hadm_id.isin(ids_with_nn_doses_d1), "received_n_narc_day1?"] = 1
    df = pd.merge(df, d1_nn_doses, on="hadm_id", how="left")

    df["received_n_narc_day2?"] = 0
    df.loc[df.hadm_id.isin(ids_with_nn_doses_d2), "received_n_narc_day2?"] = 1
    df = pd.merge(df, d2_nn_doses, on="hadm_id", how="left")

    df["received_n_narc_day3?"] = 0
    df.loc[df.hadm_id.isin(ids_with_nn_doses_d3), "received_n_narc_day3?"] = 1
    df = pd.merge(df, d3_nn_doses, on="hadm_id", how="left")

    df["received_n_narc_day4?"] = 0
    df.loc[df.hadm_id.isin(ids_with_nn_doses_d4), "received_n_narc_day4?"] = 1
    df = pd.merge(df, d4_nn_doses, on="hadm_id", how="left")

    df["received_n_narc_day5?"] = 0
    df.loc[df.hadm_id.isin(ids_with_nn_doses_d5), "received_n_narc_day5?"] = 1
    df = pd.merge(df, d5_nn_doses, on="hadm_id", how="left")

    df = pd.merge(df, total_ad_nn_dose, on="hadm_id", how="left")
    # note: to simplify, can just merge non-narc dosage values with df and set 0 to nan

    # average nn doses
    df["ad-average-nnd"] = round(df["ad-total-orders"] / df["admit_duration(days)"], 2)

    # days in hospital column
    df["days-in-hosp"] = df["admit_duration(days)"] + 1

    # received aspirin per admit day
    # aspirin dose values
    aspirin_orders_df = pd.read_csv("./data/processed/per-patient-per-day-aspirin-order-values.csv")
    as_doses_df_day_1 = aspirin_orders_df[aspirin_orders_df["day1"] > 0]
    as_doses_df_day_2 = aspirin_orders_df[aspirin_orders_df["day2"] > 0]
    as_doses_df_day_3 = aspirin_orders_df[aspirin_orders_df["day3"] > 0]
    as_doses_df_day_4 = aspirin_orders_df[aspirin_orders_df["day4"] > 0]
    as_doses_df_day_5 = aspirin_orders_df[aspirin_orders_df["day5"] > 0]
    ids_with_as_doses_d1 = as_doses_df_day_1["hadm_id"].unique()
    d1_as_doses = as_doses_df_day_1[["hadm_id", "day1"]]
    d1_as_doses.columns = ["hadm_id", "d1_ad"]

    ids_with_as_doses_d2 = as_doses_df_day_2["hadm_id"].unique()
    d2_as_doses = as_doses_df_day_2[["hadm_id", "day2"]]
    d2_as_doses.columns = ["hadm_id", "d2_ad"]

    ids_with_as_doses_d3 = as_doses_df_day_3["hadm_id"].unique()
    d3_as_doses = as_doses_df_day_3[["hadm_id", "day3"]]
    d3_as_doses.columns = ["hadm_id", "d3_ad"]

    ids_with_as_doses_d4 = as_doses_df_day_4["hadm_id"].unique()
    d4_as_doses = as_doses_df_day_4[["hadm_id", "day4"]]
    d4_as_doses.columns = ["hadm_id", "d4_ad"]

    ids_with_as_doses_d5 = as_doses_df_day_5["hadm_id"].unique()
    d5_as_doses = as_doses_df_day_5[["hadm_id", "day5"]]
    d5_as_doses.columns = ["hadm_id", "d5_ad"]

    total_ad_as_dose = aspirin_orders_df[["hadm_id", "aspirin-total-dose"]]

    df["received_asp_day1?"] = 0
    df.loc[df.hadm_id.isin(ids_with_as_doses_d1), "received_asp_day1?"] = 1
    df = pd.merge(df, d1_as_doses, on="hadm_id", how="left")
    #
    df["received_asp_day2?"] = 0
    df.loc[df.hadm_id.isin(ids_with_as_doses_d2), "received_asp_day2?"] = 1
    df = pd.merge(df, d2_as_doses, on="hadm_id", how="left")

    df["received_asp_day3?"] = 0
    df.loc[df.hadm_id.isin(ids_with_as_doses_d3), "received_asp_day3?"] = 1
    df = pd.merge(df, d3_as_doses, on="hadm_id", how="left")

    df["received_asp_day4?"] = 0
    df.loc[df.hadm_id.isin(ids_with_as_doses_d4), "received_asp_day4?"] = 1
    df = pd.merge(df, d4_as_doses, on="hadm_id", how="left")

    df["received_asp_day5?"] = 0
    df.loc[df.hadm_id.isin(ids_with_as_doses_d5), "received_asp_day5?"] = 1
    df = pd.merge(df, d5_as_doses, on="hadm_id", how="left")

    df = pd.merge(df, total_ad_as_dose, on="hadm_id", how="left")
    # note: to simplify, can just merge aspirin dosage values with df and set 0 to nan

    # average aspirin orders
    df["ad-average-asd"] = round(df["aspirin-total-dose"] / df["admit_duration(days)"], 2)

    # process statin orders
    df = drug_per_day_patient_features(df,
                                       filename="./data/processed/per-patient-per-day-statin-order-values.csv",
                                       drug_class="statin")

    # # process anti-platelets
    df = drug_per_day_patient_features(df,
                                       filename="./data/processed/per-patient-per-day-anti-platelet-order-values.csv",
                                       drug_class="anti-platelet")

    # process ace-inhibitors
    df = drug_per_day_patient_features(df,
                                       filename="./data/processed/per-patient-per-day-ace-inhibitor-order-values.csv",
                                       drug_class="ace-inhibitor")

    # process beta-blockers
    df = drug_per_day_patient_features(df,
                                       filename="./data/processed/per-patient-per-day-beta-blocker-order-values.csv",
                                       drug_class="beta-blocker")
    # df = df.drop(df.filter(regex='Unnamed').columns, axis=1)
    print("df.head() = ", df.head(), df.columns.tolist(), len(df))
    df.to_csv("./data/feature-files/ami_patients_features_master.csv", index=False)


def add_pain_therapy_type_features():
    """

    :return:
    """
    features_file = "./data/feature-files/ami_patients_features_master.csv"
    features_df = pd.read_csv(features_file)

    for i in range(1, 6):
        received_pain_med_day_i = features_df[(features_df["received_narc_day" + str(i) + "?"] == 1) |
                                               (features_df["received_n_narc_day" + str(i) + "?"] == 1)]
        ids_received_pain_med = received_pain_med_day_i["hadm_id"].unique()
        received_narc_only_day_i = features_df[(features_df["received_narc_day" + str(i) + "?"] == 1) &
                                               (features_df["received_n_narc_day" + str(i) + "?"] == 0)]
        ids_received_narcs_only = received_narc_only_day_i["hadm_id"].unique()
        received_non_narc_only_day_i = features_df[(features_df["received_narc_day" + str(i) + "?"] == 0) &
                                                   (features_df["received_n_narc_day" + str(i) + "?"] == 1)]
        ids_received_non_narcs_only = received_non_narc_only_day_i["hadm_id"].unique()
        received_combined_narc_non_narc_day_i = features_df[(features_df["received_narc_day" + str(i) + "?"] == 1) &
                                                            (features_df["received_n_narc_day" + str(i) + "?"] == 1)]
        ids_received_combined_therapy = received_combined_narc_non_narc_day_i["hadm_id"].unique()
        features_df["received_pain_med_day" + str(i) + "?"] = 0
        features_df.loc[features_df.hadm_id.isin(ids_received_pain_med), "received_pain_med_day" + str(i) + "?"] = 1
        features_df["received_narc_only_day" + str(i) + "?"] = 0
        features_df.loc[features_df.hadm_id.isin(ids_received_narcs_only), "received_narc_only_day" + str(i) + "?"] = 1
        features_df["received_n_narc_only_day" + str(i) + "?"] = 0
        features_df.loc[features_df.hadm_id.isin(ids_received_non_narcs_only),
                        "received_n_narc_only_day" + str(i) + "?"] = 1
        features_df["received_combined_therapy_day" + str(i) + "?"] = 0
        features_df.loc[features_df.hadm_id.isin(ids_received_combined_therapy),
                        "received_combined_therapy_day" + str(i) + "?"] = 1
    print("df.head() = ", features_df.head(), features_df.columns.tolist(), len(features_df))
    features_df.to_csv("./data/feature-files/ami_patients_features_master.csv", index=False)


# if __name__ == '__main__':
    # process_narcotic_prescriptions()
    # process_non_narcotic_prescriptions()
    # process_aspirin_prescriptions()
    # process_drug_prescriptions(drug_class="statin")
    # process_drug_prescriptions(drug_class="beta-blocker")
    # process_drug_prescriptions(drug_class="ace-inhibitor")
    # process_drug_prescriptions(drug_class="anti-platelet")
    # add_drug_features_in_patient_file()
    # add_pain_therapy_type_features()
