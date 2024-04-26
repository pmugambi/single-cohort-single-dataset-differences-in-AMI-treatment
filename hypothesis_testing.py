import pandas as pd
import scipy.stats as stats
from scipy.stats import fisher_exact, barnard_exact

# non-white ethnicity values
non_white_ethnicities = ["asian", "black/african american", "black/cape verdean", "hispanic or latino",
                         "middle eastern", "multi race ethnicity", "other", "white - brazilian"]


def count_admitted_at_day(features_df, by, day=1):
    """
    Counts the number of patients hospitalized until day=day. Helper function for the first-N days of hospitalization
    analyses
    :param features_df: patients' feature file containing patients' details, including length of hospitalization
    :param by: attribute to stratify patients by. values are 'sex', 'race'. and 'age'
    :param day: day of admission for which numbers should be calculated
    :return: a list with 2 values, the number of patients in the majority and minority groups. For sex,
    [#male, #famele], for race, [#White, #nonWhite], for age [#aged20-39, #aged40-50]
    """
    adm_duration_df = features_df[features_df["admit_duration(days)"] >= day]

    maj_count = None
    min_count = None

    if by == "sex":
        maj_count = len(adm_duration_df[adm_duration_df["gender"] == "m"])
        min_count = len(adm_duration_df[adm_duration_df["gender"] == "f"])

    if by == "race":
        maj_count = len(adm_duration_df[adm_duration_df["ethnicity"] == "white"])
        min_count = len(adm_duration_df[adm_duration_df["ethnicity"].isin(non_white_ethnicities)])

    if by == "age":
        maj_count = len(adm_duration_df[adm_duration_df["age"] > 39])
        min_count = len(adm_duration_df[adm_duration_df["age"] <= 39])

    return [maj_count, min_count]


def compute_proportions_per_treatment(features_df, s_attribute, treatment_key, day=1):
    """
    Generates a contingency table of number of patients that received or did not receive a specific treatment
    <treatment_key>, when patients are stratified by <s_attribute> for patients admitted until at least day <day>
    :param features_df: patients' features file
    :param s_attribute: attribute to stratify patients by
    :param treatment_key: the name of the treatment for which proportions are to be computed
    :param day: day number
    :return: a contingency table of numbers of those who received and didn't receive treatment
    """
    maj_rt = None
    maj_dnrt = None
    min_rt = None
    min_dnrt = None

    rt = features_df[(features_df[treatment_key] == 1) & (features_df["admit_duration(days)"] >= day)]
    # returns the dataframe of patients who received the treatment under evaluation, if they were hospitalized for
    # at least <day> days

    maj_total, min_total = count_admitted_at_day(by=s_attribute, features_df=features_df, day=day)

    if s_attribute == "sex":
        maj_rt = len(rt[rt["gender"] == "m"])
        min_rt = len(rt[rt["gender"] == "f"])
        maj_dnrt = maj_total - maj_rt
        min_dnrt = min_total - min_rt

    elif s_attribute == "race":
        maj_rt = len(rt[rt["ethnicity"] == "white"])
        min_rt = len(rt[rt["ethnicity"].isin(non_white_ethnicities)])
        maj_dnrt = maj_total - maj_rt
        min_dnrt = min_total - min_rt

    elif s_attribute == "age":
        maj_rt = len(rt[rt["age"] > 39])
        min_rt = len(rt[rt["age"] <= 39])
        maj_dnrt = maj_total - maj_rt
        min_dnrt = min_total - min_rt

    return [[maj_rt, maj_dnrt], [min_rt, min_dnrt]]  # returns contingency table values, beginning with the majority
    # group


def create_sub_treatment_proportions_list(treatment_key, s_attribute, features_df):
    """
    Generates a contingency table of patients that received a specific treatment for the first five days of
    hospitalization. Iterates over days 1-5 and uses compute_proportions_per_treatment() above to generate the
    contingency table for each of the 5 days when patients are stratified by <s_attribute>
    :param features_df: patients' features file
    :param treatment_key: name of the treatment for which proportions are being computed
    :param s_attribute: the attribute by which to stratify patients
    :return: a list of 6 contingency tables. First 5 are tables for each of the first 5 days of hospitalization,
    and the 6th is a summation of the values across the first 5 days. We sum them to increase the sample size and
    consequently the power of the test
    """
    props = []
    for i in range(0, 5):
        key = treatment_key + str(i + 1) + "?"
        i_list = compute_proportions_per_treatment(s_attribute=s_attribute, treatment_key=key,
                                                   features_df=features_df, day=i)
        props.append(i_list)
    g1_r = 0
    g1_dnr = 0
    g2_r = 0
    g2_dnr = 0
    for i in range(0, 5):
        g1_r += props[i][0][0]
        g1_dnr += props[i][0][1]
        g2_r += props[i][1][0]
        g2_dnr += props[i][1][1]
    days_sum_props = [[g1_r, g1_dnr], [g2_r, g2_dnr]]
    props.append(days_sum_props)
    return props


def compute_proportions(treatment_group, duration, s_attribute, features_df):
    """
    Computes the proportions that received treatment for the entire hospitalization and first 5 days of hospitalization
    for all treatment therapies. Uses create_sub_treatment_proportions_list() and compute_proportions_per_treatment()
    defined above.
    :param treatment_group: whether the analysis is for analgesia or AMI-related drugs
    :param duration: whether it's for the entire hospitalization or the first 5 days
    :param s_attribute: attribute by which to stratify the patients
    :param features_df: patients' features file
    :return: a dictionary of the contingency tables for each of the treatment therapies in the specified treatment_group
    """
    if duration == "first-five-days":
        if treatment_group == "pain-therapy":
            # compute proportions for all pain treatment therapies
            pain_med_props = create_sub_treatment_proportions_list(treatment_key="received_pain_med_day",
                                                                   s_attribute=s_attribute, features_df=features_df)
            # any-narcotics:
            narc_props = create_sub_treatment_proportions_list(treatment_key="received_narc_day",
                                                               s_attribute=s_attribute, features_df=features_df)
            non_narc_props = create_sub_treatment_proportions_list(treatment_key="received_n_narc_day",
                                                                   s_attribute=s_attribute, features_df=features_df)
            narc_only_props = create_sub_treatment_proportions_list(treatment_key="received_narc_only_day",
                                                                    s_attribute=s_attribute, features_df=features_df)
            non_narc_only_props = create_sub_treatment_proportions_list(treatment_key="received_n_narc_only_day",
                                                                        s_attribute=s_attribute,
                                                                        features_df=features_df)
            combined_treatment_props = create_sub_treatment_proportions_list(
                treatment_key="received_combined_therapy_day", s_attribute=s_attribute, features_df=features_df)
            return {
                "pain-med-props": pain_med_props,
                "narc-props": narc_props,
                "non-narc-props": non_narc_props,
                "narc-only-props": narc_only_props,
                "non-narc-only-props": non_narc_only_props,
                "combined-treatment-props": combined_treatment_props
            }

        elif treatment_group == "ami-therapy":
            # compute proportions for each AMI-drug therapy
            aspirin_props = create_sub_treatment_proportions_list(treatment_key="received_asp_day",
                                                                  s_attribute=s_attribute, features_df=features_df)
            non_aspirin_ap_props = create_sub_treatment_proportions_list(
                treatment_key="anti-platelet_day", s_attribute=s_attribute, features_df=features_df)
            bb_props = create_sub_treatment_proportions_list(treatment_key="beta-blocker_day",
                                                             s_attribute=s_attribute, features_df=features_df)
            ai_props = create_sub_treatment_proportions_list(treatment_key="ace-inhibitor_day",
                                                             s_attribute=s_attribute, features_df=features_df)
            statin_props = create_sub_treatment_proportions_list(treatment_key="statin_day",
                                                                 s_attribute=s_attribute, features_df=features_df)
            return {
                "aspirin-props": aspirin_props,
                "non-aspirin-anti-platelet-props": non_aspirin_ap_props,
                "beta-blocker-props": bb_props,
                "ace-inhibitor-props": ai_props,
                "statin-props": statin_props
            }
        else:
            print("This analysis evaluates differences in pain and AMI drug therapies only.\n "
                  "Acceptable input values are: 'pain-therapy' and 'ami-therapy'")
    elif duration == "entire-admission-period":
        if treatment_group == "pain-therapy":
            any_pain_med_props = compute_proportions_per_treatment(treatment_key="received_pain_med?",
                                                                   s_attribute=s_attribute, features_df=features_df)
            narc_props = compute_proportions_per_treatment(treatment_key="received_narcotic?", s_attribute=s_attribute,
                                                           features_df=features_df)
            return {
                "pain-medication-props": any_pain_med_props,
                "narcotic-props": narc_props,
            }
        elif treatment_group == "ami-therapy":
            aspirin_props = compute_proportions_per_treatment(treatment_key="aspirin?",
                                                              s_attribute=s_attribute, features_df=features_df)
            non_aspirin_ap_props = compute_proportions_per_treatment(treatment_key="received_anti-platelet?",
                                                                     s_attribute=s_attribute, features_df=features_df)
            bb_props = compute_proportions_per_treatment(treatment_key="received_beta-blocker?",
                                                         s_attribute=s_attribute, features_df=features_df)
            ai_props = compute_proportions_per_treatment(treatment_key="received_ace-inhibitor?",
                                                         s_attribute=s_attribute, features_df=features_df)
            statin_props = compute_proportions_per_treatment(treatment_key="received_statin?",
                                                             s_attribute=s_attribute, features_df=features_df)
            return {
                "aspirin-props": aspirin_props,
                "non-aspirin-anti-platelet-props": non_aspirin_ap_props,
                "beta-blocker-props": bb_props,
                "ace-inhibitor-props": ai_props,
                "statin-props": statin_props
            }
        elif treatment_group == "procedures":
            any_procedure_props = compute_proportions_per_treatment(treatment_key="received_procedure?",
                                                                    s_attribute=s_attribute, features_df=features_df)

            surgical_procedures_props = compute_proportions_per_treatment(treatment_key="surgical_procedure?",
                                                                          s_attribute=s_attribute,
                                                                          features_df=features_df)
            non_surgical_procedures_props = compute_proportions_per_treatment(treatment_key="non_surgical_procedure?",
                                                                              s_attribute=s_attribute,
                                                                              features_df=features_df)
            other_procedures_props = compute_proportions_per_treatment(treatment_key="other_procedure?",
                                                                       s_attribute=s_attribute, features_df=features_df)
            return {
                "procedures-props": any_procedure_props,
                "surgical-procedures-props": surgical_procedures_props,
                "non-surgical-procedures-props": non_surgical_procedures_props,
                "other-procedures-props": other_procedures_props
            }
        else:
            print("This analysis evaluates differences in pain and AMI drug therapies only.\n"
                  "Acceptable input values are: 'pain-therapy', 'ami-therapy', and 'procedures'")
    else:
        print("This analysis evaluates differences in pain and AMI drug therapies only.\n"
              "A duration of evaluation is a required input.\n"
              "Acceptable input values are: 'entire-admission-period' and 'first-five-days'")


def compute_p_values(contingency_table, test_name="barnard", alternative="two-sided"):
    """
    Computes p-value for proportions (passed as a contingency table) using exact test.
    :return: a dictionary containing the p-value and it's associated statistics
    """

    if test_name == "fisher":
        oddsr, p = fisher_exact(contingency_table, alternative=alternative)
        return {"odds-ratio": oddsr, "p-value": p}

    elif test_name == "barnard":
        res = barnard_exact(contingency_table, alternative=alternative)
        return {"statistic": res.statistic, "p-value": res.pvalue}

    else:
        print("Currently, this analysis is only tested using either 'Fisher' or 'Barnard' test \n"
              "Allowed inputs are 'fisher' and 'barnard' ")


def analgesic_differences_entire_los():
    """
    Obtains the p-values for orders of analgesic treatments for the entire hospitalization
    :return: nothing. all results are written in files in ./data/results/entire-admission/
    """
    all_patients_df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    surgical_procedure_df = all_patients_df[all_patients_df["surgical_procedure?"] == 1]
    non_surgical_or_other_procedure_df = all_patients_df[(all_patients_df["non_surgical_procedure?"] == 1) &
                                                     (all_patients_df["surgical_procedure?"] == 0)]

    entire_admission_rows = []

    # entire duration, analgesics
    sex_all_analgesics_props = compute_proportions(s_attribute="sex",
                                                   treatment_group="pain-therapy",
                                                   duration='entire-admission-period',
                                                   features_df=all_patients_df)
    race_all_analgesics_props = compute_proportions(s_attribute="race",
                                                    treatment_group="pain-therapy",
                                                    duration='entire-admission-period',
                                                    features_df=all_patients_df)
    age_all_analgesics_props = compute_proportions(s_attribute="age",
                                                   treatment_group="pain-therapy",
                                                   duration='entire-admission-period',
                                                   features_df=all_patients_df)
    sex_surg_procs_analgesics_props = compute_proportions(s_attribute="sex",
                                                          treatment_group="pain-therapy",
                                                          duration='entire-admission-period',
                                                          features_df=surgical_procedure_df)
    race_surg_procs_analgesics_props = compute_proportions(s_attribute="race",
                                                           treatment_group="pain-therapy",
                                                           duration='entire-admission-period',
                                                           features_df=surgical_procedure_df)
    age_surg_procs_analgesics_props = compute_proportions(s_attribute="age",
                                                          treatment_group="pain-therapy",
                                                          duration='entire-admission-period',
                                                          features_df=surgical_procedure_df)
    sex_non_surg_procs_analgesics_props = compute_proportions(s_attribute="sex",
                                                              treatment_group="pain-therapy",
                                                              duration='entire-admission-period',
                                                              features_df=non_surgical_or_other_procedure_df)
    race_non_surg_procs_analgesics_props = compute_proportions(s_attribute="race",
                                                               treatment_group="pain-therapy",
                                                               duration='entire-admission-period',
                                                               features_df=non_surgical_or_other_procedure_df)
    age_non_surg_procs_analgesics_props = compute_proportions(s_attribute="age",
                                                              treatment_group="pain-therapy",
                                                              duration='entire-admission-period',
                                                              features_df=non_surgical_or_other_procedure_df)

    entire_admission_rows.append(["all-analgesics", None, "sex",
                                  compute_p_values(sex_all_analgesics_props["pain-medication-props"])["p-value"]])
    entire_admission_rows.append(["all-analgesics", None, "race",
                                  compute_p_values(race_all_analgesics_props["pain-medication-props"])["p-value"]])
    entire_admission_rows.append(["all-analgesics", None, "age",
                                  compute_p_values(age_all_analgesics_props["pain-medication-props"])["p-value"]])
    entire_admission_rows.append(["all-analgesics", "surgical-procedures", "sex",
                                  compute_p_values(sex_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["all-analgesics", "surgical-procedures", "race",
                                  compute_p_values(race_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["all-analgesics", "surgical-procedures", "age",
                                  compute_p_values(age_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["all-analgesics", "non-surgical-procedures", "sex",
                                  compute_p_values(sex_non_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["all-analgesics", "non-surgical-procedures", "race",
                                  compute_p_values(race_non_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["all-analgesics", "non-surgical-procedures", "age",
                                  compute_p_values(age_non_surg_procs_analgesics_props["pain-medication-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", None, "sex",
                                  compute_p_values(sex_all_analgesics_props["narcotic-props"])["p-value"]])
    entire_admission_rows.append(["narcotics", None, "race",
                                  compute_p_values(race_all_analgesics_props["narcotic-props"])["p-value"]])
    entire_admission_rows.append(["narcotics", None, "age",
                                  compute_p_values(age_all_analgesics_props["narcotic-props"])["p-value"]])
    entire_admission_rows.append(["narcotics", "surgical-procedures", "sex",
                                  compute_p_values(sex_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", "surgical-procedures", "race",
                                  compute_p_values(race_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", "surgical-procedures", "age",
                                  compute_p_values(age_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", "non-surgical-procedures", "sex",
                                  compute_p_values(sex_non_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", "non-surgical-procedures", "race",
                                  compute_p_values(race_non_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])
    entire_admission_rows.append(["narcotics", "non-surgical-procedures", "age",
                                  compute_p_values(age_non_surg_procs_analgesics_props["narcotic-props"]
                                                   )["p-value"]])

    df = pd.DataFrame(data=entire_admission_rows, columns=["treatment", "conditioned-on", "attribute", "p-value"])
    df.to_csv("./data/results/entire-admission/p-values/differences-in-proportions-that-received-analgesia.csv",
              index=False)


def aspirin_differences_entire_los():
    """
    Obtains the p-values for orders of aspirin for the entire hospitalization
    :return: nothing. all results are written in files in ./data/results/entire-admission/
    """
    all_patients_df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    stemi_only_df = all_patients_df[all_patients_df["non_stemi?"] == 0]
    nstemi_only_df = all_patients_df[all_patients_df["non_stemi?"] == 1]

    rows = []

    # entire duration, analgesics
    sex_aspirin_all_props = compute_proportions(s_attribute="sex",
                                                treatment_group="ami-therapy",
                                                duration='entire-admission-period',
                                                features_df=all_patients_df)
    race_aspirin_all_props = compute_proportions(s_attribute="race",
                                                 treatment_group="ami-therapy",
                                                 duration='entire-admission-period',
                                                 features_df=all_patients_df)
    age_aspirin_all_props = compute_proportions(s_attribute="age",
                                                treatment_group="ami-therapy",
                                                duration='entire-admission-period',
                                                features_df=all_patients_df)
    sex_aspirin_stemi_props = compute_proportions(s_attribute="sex",
                                                  treatment_group="ami-therapy",
                                                  duration='entire-admission-period',
                                                  features_df=stemi_only_df)
    race_aspirin_stemi_props = compute_proportions(s_attribute="race",
                                                   treatment_group="ami-therapy",
                                                   duration='entire-admission-period',
                                                   features_df=stemi_only_df)
    age_aspirin_stemi_props = compute_proportions(s_attribute="age",
                                                  treatment_group="ami-therapy",
                                                  duration='entire-admission-period',
                                                  features_df=stemi_only_df)
    sex_aspirin_nstemi_props = compute_proportions(s_attribute="sex",
                                                   treatment_group="ami-therapy",
                                                   duration='entire-admission-period',
                                                   features_df=nstemi_only_df)
    race_aspirin_nstemi_props = compute_proportions(s_attribute="race",
                                                    treatment_group="ami-therapy",
                                                    duration='entire-admission-period',
                                                    features_df=nstemi_only_df)
    age_aspirin_nstemi_props = compute_proportions(s_attribute="age",
                                                   treatment_group="ami-therapy",
                                                   duration='entire-admission-period',
                                                   features_df=nstemi_only_df)

    rows.append(["aspirin", None, "sex",
                 compute_p_values(sex_aspirin_all_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", None, "race",
                 compute_p_values(race_aspirin_all_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", None, "age",
                 compute_p_values(age_aspirin_all_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "stemi-diagnosis", "sex",
                 compute_p_values(sex_aspirin_stemi_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "stemi-diagnosis", "race",
                 compute_p_values(race_aspirin_stemi_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "stemi-diagnosis", "age",
                 compute_p_values(age_aspirin_stemi_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "nstemi-diagnosis", "sex",
                 compute_p_values(sex_aspirin_nstemi_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "nstemi-diagnosis", "race",
                 compute_p_values(race_aspirin_nstemi_props["aspirin-props"])["p-value"]])
    rows.append(["aspirin", "nstemi-diagnosis", "age",
                 compute_p_values(age_aspirin_nstemi_props["aspirin-props"])["p-value"]])
    df = pd.DataFrame(data=rows, columns=["treatment", "conditioned-on", "attribute", "p-value"])
    df.to_csv("./data/results/entire-admission/p-values/differences-in-proportions-that-received-aspirin.csv",
              index=False)


def procedures_differences_entire_los():
    """
    Obtains the p-values for medical procedures performed at any point of hospitalization
    :return: nothing. all results are written in files in ./data/results/entire-admission/
    """
    all_patients_df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    stemi_only_df = all_patients_df[all_patients_df["non_stemi?"] == 0]
    nstemi_only_df = all_patients_df[all_patients_df["non_stemi?"] == 1]

    rows = []

    # entire duration, any procedures
    sex_any_procedure_props = compute_proportions(s_attribute="sex",
                                                  treatment_group="procedures",
                                                  duration='entire-admission-period',
                                                  features_df=all_patients_df)
    race_any_procedure_props = compute_proportions(s_attribute="race",
                                                   treatment_group="procedures",
                                                   duration='entire-admission-period',
                                                   features_df=all_patients_df)
    age_any_procedure_props = compute_proportions(s_attribute="age",
                                                  treatment_group="procedures",
                                                  duration='entire-admission-period',
                                                  features_df=all_patients_df)
    sex_any_procedure_stemi_props = compute_proportions(s_attribute="sex",
                                                        treatment_group="procedures",
                                                        duration='entire-admission-period',
                                                        features_df=stemi_only_df)
    race_any_procedure_stemi_props = compute_proportions(s_attribute="race",
                                                         treatment_group="procedures",
                                                         duration='entire-admission-period',
                                                         features_df=stemi_only_df)
    age_any_procedure_stemi_props = compute_proportions(s_attribute="age",
                                                        treatment_group="procedures",
                                                        duration='entire-admission-period',
                                                        features_df=stemi_only_df)
    sex_any_procedure_nstemi_props = compute_proportions(s_attribute="sex",
                                                         treatment_group="procedures",
                                                         duration='entire-admission-period',
                                                         features_df=nstemi_only_df)
    race_any_procedure_nstemi_props = compute_proportions(s_attribute="race",
                                                          treatment_group="procedures",
                                                          duration='entire-admission-period',
                                                          features_df=nstemi_only_df)
    age_any_procedure_nstemi_props = compute_proportions(s_attribute="age",
                                                         treatment_group="procedures",
                                                         duration='entire-admission-period',
                                                         features_df=nstemi_only_df)

    rows.append(["any-procedure", None, "sex",
                 compute_p_values(sex_any_procedure_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", None, "race",
                 compute_p_values(race_any_procedure_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", None, "age",
                 compute_p_values(age_any_procedure_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "stemi-diagnosis", "sex",
                 compute_p_values(sex_any_procedure_stemi_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "stemi-diagnosis", "race",
                 compute_p_values(race_any_procedure_stemi_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "stemi-diagnosis", "age",
                 compute_p_values(age_any_procedure_stemi_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "nstemi-diagnosis", "sex",
                 compute_p_values(sex_any_procedure_nstemi_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "nstemi-diagnosis", "race",
                 compute_p_values(race_any_procedure_nstemi_props["procedures-props"])["p-value"]])
    rows.append(["any-procedure", "nstemi-diagnosis", "age",
                 compute_p_values(age_any_procedure_nstemi_props["procedures-props"])["p-value"]])

    rows.append(["surgical-procedure", None, "sex",
                 compute_p_values(sex_any_procedure_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", None, "race",
                 compute_p_values(race_any_procedure_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", None, "age",
                 compute_p_values(age_any_procedure_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "stemi-diagnosis", "sex",
                 compute_p_values(sex_any_procedure_stemi_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "stemi-diagnosis", "race",
                 compute_p_values(race_any_procedure_stemi_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "stemi-diagnosis", "age",
                 compute_p_values(age_any_procedure_stemi_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "nstemi-diagnosis", "sex",
                 compute_p_values(sex_any_procedure_nstemi_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "nstemi-diagnosis", "race",
                 compute_p_values(race_any_procedure_nstemi_props["surgical-procedures-props"])["p-value"]])
    rows.append(["surgical-procedure", "nstemi-diagnosis", "age",
                 compute_p_values(age_any_procedure_nstemi_props["surgical-procedures-props"])["p-value"]])
    df = pd.DataFrame(data=rows, columns=["treatment", "conditioned-on", "attribute", "p-value"])
    df.to_csv("./data/results/entire-admission/p-values/differences-in-proportions-that-received-procedures.csv",
              index=False)


def generate_first_five_days_differences(treatment_group, treatment_results_vars, treatment_names, save_name):
    """
    Computes the p-values of proportions that received treatment for the first 5 days of hospitalization for all
    treatment therapies. Uses compute_proportions() defined above.
    :param treatment_group: whether the analysis of for analgesia or AMI-related drugs
    :param treatment_results_vars: treatment variable names from the obj returned by compute_proportions()
    :param treatment_names: the names of the various treatments. used to record the final p-values in the csv files
    :param save_name: path and filename to save the p-values
    :return: nothing. results written to files that are saved in ./data/results/first-five-days/
    """
    all_patients_df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    c_shock_only_df = all_patients_df[all_patients_df["c-shock?"] == 1]
    no_c_shock_only_df = all_patients_df[all_patients_df["c-shock?"] == 0]
    sex_props = compute_proportions(s_attribute="sex",
                                    treatment_group=treatment_group,
                                    duration='first-five-days',
                                    features_df=all_patients_df)
    race_props = compute_proportions(s_attribute="race",
                                     treatment_group=treatment_group,
                                     duration='first-five-days',
                                     features_df=all_patients_df)
    age_props = compute_proportions(s_attribute="age",
                                    treatment_group=treatment_group,
                                    duration='first-five-days',
                                    features_df=all_patients_df)

    c_shock_sex_props = compute_proportions(s_attribute="sex",
                                            treatment_group=treatment_group,
                                            duration='first-five-days',
                                            features_df=c_shock_only_df)
    c_shock_race_props = compute_proportions(s_attribute="race",
                                             treatment_group=treatment_group,
                                             duration='first-five-days',
                                             features_df=c_shock_only_df)
    c_shock_age_props = compute_proportions(s_attribute="age",
                                            treatment_group=treatment_group,
                                            duration='first-five-days',
                                            features_df=c_shock_only_df)

    no_c_shock_sex_props = compute_proportions(s_attribute="sex",
                                               treatment_group=treatment_group,
                                               duration='first-five-days',
                                               features_df=no_c_shock_only_df)
    no_c_shock_race_props = compute_proportions(s_attribute="race",
                                                treatment_group=treatment_group,
                                                duration='first-five-days',
                                                features_df=no_c_shock_only_df)
    no_c_shock_age_props = compute_proportions(s_attribute="age",
                                               treatment_group=treatment_group,
                                               duration='first-five-days',
                                               features_df=no_c_shock_only_df)
    days = range(0, 6)
    rows = []
    for a in range(len(treatment_results_vars)):
        treatment_res_var_name = treatment_results_vars[a]
        treatment_name = treatment_names[a]
        therapy_sex_all_p_values = [compute_p_values(sex_props[treatment_res_var_name][i])["p-value"]
                                    for i in days]
        therapy_sex_c_shock_p_values = [compute_p_values(c_shock_sex_props[treatment_res_var_name][i])
                                        ["p-value"] for i in days]
        therapy_sex_no_c_shock_p_values = [compute_p_values(no_c_shock_sex_props[treatment_res_var_name][i])
                                           ["p-value"] for i in days]
        # race
        therapy_race_all_p_values = [compute_p_values(race_props[treatment_res_var_name][i])["p-value"]
                                     for i in days]
        therapy_race_c_shock_p_values = [compute_p_values(c_shock_race_props[treatment_res_var_name][i])
                                         ["p-value"] for i in days]
        therapy_race_no_c_shock_p_values = [compute_p_values(no_c_shock_race_props[treatment_res_var_name][i])
                                            ["p-value"] for i in days]
        # age
        therapy_age_all_p_values = [compute_p_values(age_props[treatment_res_var_name][i])["p-value"]
                                    for i in days]
        therapy_age_c_shock_p_values = [compute_p_values(c_shock_age_props[treatment_res_var_name][i])
                                        ["p-value"] for i in days]
        therapy_age_no_c_shock_p_values = [compute_p_values(no_c_shock_age_props[treatment_res_var_name][i])
                                           ["p-value"] for i in days]

        for i in days:
            day = i + 1
            if day == 6:
                day = "1-5"
            # sex - no conditioning
            rows.append([treatment_name, day, "sex", None] + extract_majority_minority_proportions(
                sex_props[treatment_res_var_name][i]) + [therapy_sex_all_p_values[i]])
            # sex - conditioning on c-shock diagnosis positive
            rows.append([treatment_name, day, "sex", "c-shock"] + extract_majority_minority_proportions(
                c_shock_sex_props[treatment_res_var_name][i]) + [therapy_sex_c_shock_p_values[i]])
            # sex - conditioning on c-shock diagnosis negative
            rows.append([treatment_name, day, "sex", "no-c-shock"] + extract_majority_minority_proportions(
                no_c_shock_sex_props[treatment_res_var_name][i]) + [therapy_sex_no_c_shock_p_values[i]])
            # race - no conditioning
            rows.append([treatment_name, day, "race", None] + extract_majority_minority_proportions(
                race_props[treatment_res_var_name][i]) + [therapy_race_all_p_values[i]])
            # race - conditioning on c-shock diagnosis positive
            rows.append([treatment_name, day, "race", "c-shock"] + extract_majority_minority_proportions(
                c_shock_race_props[treatment_res_var_name][i]) + [therapy_race_c_shock_p_values[i]])
            # race - conditioning on c-shock diagnosis negative
            rows.append([treatment_name, day, "race", "no-c-shock"] + extract_majority_minority_proportions(
                no_c_shock_race_props[treatment_res_var_name][i]) + [therapy_race_no_c_shock_p_values[i]])
            # age - no conditioning
            rows.append([treatment_name, day, "age", None] + extract_majority_minority_proportions(
                age_props[treatment_res_var_name][i]) + [therapy_age_all_p_values[i]])
            # race - conditioning on c-shock diagnosis positive
            rows.append([treatment_name, day, "age", "c-shock"] + extract_majority_minority_proportions(
                c_shock_age_props[treatment_res_var_name][i]) + [therapy_age_c_shock_p_values[i]])
            # race - conditioning on c-shock diagnosis negative
            rows.append([treatment_name, day, "age", "no-c-shock"] + extract_majority_minority_proportions(
                no_c_shock_age_props[treatment_res_var_name][i]) + [therapy_age_no_c_shock_p_values[i]])

    # create and save df
    df = pd.DataFrame(data=rows,
                      columns=["treatment", "day-no", "attribute", "conditioned-on", "majority-yes", "majority-no",
                               "minority-yes", "minority-no", "p-value"])
    df.to_csv("./data/results/first-five-days/p-values/" + save_name, index=False)


def analgesic_differences_first_five_days():
    """
    Generates the p-values for all analgesic treatments for each of the first 5 days of hospitalization.
    Uses generate_first_five_days_differences() defined above.
    :return: nothing. all results written into files and saved in ./data/results/first-five-days/
    """
    treatments = ["pain-med-props", "narc-props", "non-narc-props", "narc-only-props", "non-narc-only-props",
                  "combined-treatment-props"]
    treatment_names = ["any-analgesia", "narcotics", "non-narcotics", "narcotics-only", "non-narcotics-only",
                       "combined-therapy"]
    generate_first_five_days_differences(treatment_group="pain-therapy",
                                         treatment_results_vars=treatments,
                                         treatment_names=treatment_names,
                                         save_name="differences-in-proportions-that-received-analgesia.csv")


def ami_therapies_differences_five_days():
    """
    Generates the p-values for all AMI-related treatments for each of the first 5 days of hospitalization.
    Uses generate_first_five_days_differences() defined above.
    :return: nothing. all results written into files and saved in ./data/results/first-five-days/
    """
    treatments = ["aspirin-props", "non-aspirin-anti-platelet-props", "beta-blocker-props", "ace-inhibitor-props",
                  "statin-props"]
    treatment_names = ["aspirin", "non-aspirin-anti-platelet", "beta-blocker", "ace-inhibitor", "statin"]
    generate_first_five_days_differences(treatment_group="ami-therapy", treatment_results_vars=treatments,
                                         treatment_names=treatment_names,
                                         save_name="differences-in-proportions-that-received-ami-therapies.csv")


def extract_majority_minority_proportions(props_obj):
    """
    Helper function to obtain the proportions that received/didn't receive treatment.
    :param props_obj: a list of lists containing proportions that received/didn't receive treatment for the first 5
    days of hospitalization
    :return: a list of 4 elements of number of patients who received and didn't receive treatment for the majority and
    minority subgroups when patients are stratified by sex, age, or race.
    """
    majority_r = props_obj[0][0]
    majority_dnr = props_obj[0][1]
    minority_r = props_obj[1][0]
    minority_dnr = props_obj[1][1]

    return [majority_r, majority_dnr, minority_r, minority_dnr]


def generate_differences_in_medication_dosage(df_key, key_name, sensitive_attribute):
    """
    Runs a t-test to compute p-values of differences in means of medicine dosages/order-counts for the two patient
    subgroups obtained after patients are stratified by either sex, race, or age
    :param df_key: the feature name in the patients' feature file representing the dosage/order-count for which the
    t-test is being run
    :param key_name: treatment name to use to refer to the difference in dosage/order-count name when saving the results
    :param sensitive_attribute:
    :return: a list of the treatment, averages, and the t-test p-value and associated statistic
    """
    df = pd.read_csv("./data/feature-files/ami_patients_features_master.csv")
    if sensitive_attribute == "gender":
        minority_df = df[df["gender"] == "f"]
        majority_df = df[df["gender"] == "m"]
    elif sensitive_attribute == "race":
        majority_df = df[df["ethnicity"] == "white"]
        minority_df = df[df["ethnicity"].isin(non_white_ethnicities)]
    elif sensitive_attribute == "age":
        majority_df = df[df["age"] > 39]
        minority_df = df[df["age"] <= 39]
    else:
        majority_df = None
        minority_df = None
    majority_values = majority_df[df_key].dropna()
    minority_values = minority_df[df_key].dropna()

    majority_average = sum(majority_values) / len(majority_values)
    minority_average = sum(minority_values) / len(minority_values)

    test_val = stats.ttest_ind(majority_values, minority_values)
    statistic = list(test_val)[0]
    p_value = list(test_val)[1]
    return [key_name, sensitive_attribute, p_value, statistic, majority_average, minority_average]


def dosage_or_order_count_differences_entire_los():
    """
    Generates all t-test p-values for all treatments and when patients are stratified by each of the 3 attributes.
    Uses generate_differences_in_medication_dosage() defined above.
    :return: nothing. all results written into a file and saved in ./data/results/entire-admission/
    """
    sensitive_attributes = ["gender", "race", "age"]
    keys = ["ad-average-mme", "ad-average-nnd", "ad-average-asd", "ad-average-statin",
            "ad-average-anti-platelet", "ad-average-ace-inhibitor", "ad-average-beta-blocker"]
    key_names = ["average-mme", "average-non-narc-orders", "average-aspirin-orders", "average-statin-orders",
                 "average-anti-platelet-orders", "average-ace-inhibitor-orders", "average-beta-blocker-orders"]
    ls = []
    col_names = ["feature-name", "sensitive-attribute", "p-value", "statistic", "maj_average_val", "min_average_val"]

    for sa in sensitive_attributes:
        for i in range(len(keys)):
            key = keys[i]
            key_name = key_names[i]
            ls.append(generate_differences_in_medication_dosage(df_key=key, key_name=key_name, sensitive_attribute=sa))
    df = pd.DataFrame(ls, columns=col_names)
    df.to_csv("./data/results/entire-admission/p-values/differences-in-dosage-or-orders-of-various-drugs.csv",
              index=False)


if __name__ == '__main__':
    # 1. Differences in entire admission duration
    # get differences in proportions that received analgesia
    analgesic_differences_entire_los()
    # get differences in proportions that received aspirin
    aspirin_differences_entire_los()
    # get differences in proportions that received medical procedures
    procedures_differences_entire_los()
    # 2. Differences in the first five days
    # get differences in proportions that received analgesia
    analgesic_differences_first_five_days()
    # get differences in proportions that received AMI-related drugs
    ami_therapies_differences_five_days()
    # 3. Differences in orders/dosage of various drugs during entire admission duration
    dosage_or_order_count_differences_entire_los()
