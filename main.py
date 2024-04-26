import data_processing, process_drug_orders
import hypothesis_testing
import outcome_analysis
import plotting_data

if __name__ == '__main__':
    """
    first, organize patients' procedures
    """
    print("1. Organizing patients' medical procedures ")
    data_processing.organize_patient_procedures()

    """
    second, create a patients' feature file, with patient attributes; i.e., their sex, ethnicity, ege, admission
    duration, disease severity markers (e.g., STEMI vs NSTEMI, liver/kidney issues ...), and whether they received
    analgesics
    """
    print("2. Creating a patients' features file")
    data_processing.create_patient_features_file()

    """
    now, we process disease severity data
    """
    # first, we read and modify troponin values
    print("3. Cleaning the troponin records file")
    data_processing.clean_troponin_file()
    # then we generate features troponin, creatine, and shock
    print("4. Processing disease severity markers")
    data_processing.process_disease_severity_markers()
    """
    now, we add the outcome features
    """
    print("5. Adding outcome features to the patients' features file")
    data_processing.add_outcome_features()
    """
    next, process drug orders
    """
    # separate the prescriptions file into several files for each of AMI-related drugs
    print("6. Writing clean AMI-related drugs prescriptions files")
    data_processing.write_prescriptions_files("./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    # process narcotic analgesics
    print("7. Processing narcotic orders")
    process_drug_orders.process_narcotic_prescriptions()
    # process non-narcotic analgesics
    print("8. Processing non-narcotic orders")
    process_drug_orders.process_non_narcotic_prescriptions()
    # process aspirin orders
    print("9. Processing aspirin orders")
    process_drug_orders.process_aspirin_prescriptions()
    # process statin orders
    print("10. Processing statin orders")
    process_drug_orders.process_drug_prescriptions(drug_class="statin")
    # process beta-blocker orders
    print("11. Processing beta-blocker orders")
    process_drug_orders.process_drug_prescriptions(drug_class="beta-blocker")
    # process ace-inhibitor orders
    print("12. Processing ACE-inhibitor orders")
    process_drug_orders.process_drug_prescriptions(drug_class="ace-inhibitor")
    # process non-aspirin anti-platelet orders
    print("13. Processing non-aspirin antiplatelet orders")
    process_drug_orders.process_drug_prescriptions(drug_class="anti-platelet")
    # add all drug order features to the feature file
    print("14. Adding drug features to the patients' features file")
    process_drug_orders.add_drug_features_in_patient_file()
    process_drug_orders.add_pain_therapy_type_features()
    """
    # run hypothesis tests to generate p-values
    # """
    # first obtain differences for the entire-admission duration
    print("15. Running hypothesis tests for the entire hospitalization")
    # differences in orders of analgesia
    hypothesis_testing.analgesic_differences_entire_los()
    # differences in orders of aspirin
    hypothesis_testing.aspirin_differences_entire_los()
    # differences in medical procedures
    hypothesis_testing.procedures_differences_entire_los()
    # differences in means of the dosage/order-counts
    hypothesis_testing.dosage_or_order_count_differences_entire_los()
    # next, obtain differences for the first-five days of admission
    print("16. Running hypothesis tests for the the first 5 days of hospitalization")
    # differences in orders of analgesia
    hypothesis_testing.analgesic_differences_first_five_days()
    # differences in orders of AMI-related drugs
    hypothesis_testing.ami_therapies_differences_five_days()
    """
    # fit logistic regression models to obtain association of treatment features with outcome
    # """
    print("17. Fitting logistic regression to obtain association between treatment and patient outcome")
    outcome_analysis.main()
    """
    # generate plots
    # """
    print("18. Generating plots of the results of hypothesis tests")
    plotting_data.main()

