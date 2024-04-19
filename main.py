import data_processing, process_drug_orders
import hypothesis_testing
import outcome_analysis

if __name__ == '__main__':
    """
    first, organize patients' procedures
    """
    data_processing.organize_patient_procedures()

    """
    second, create a patients' feature file, with patient attributes; i.e., their sex, ethnicity, ege, admission 
    duration, disease severity markers (e.g., STEMI vs NSTEMI, liver/kidney issues ...), and whether they received 
    analgesics
    """
    data_processing.create_patient_features_file()

    """
    now, we process disease severity data
    """
    # first, we read and modify troponin values
    data_processing.clean_troponin_file()
    # then we generate features troponin, creatine, and shock
    data_processing.process_disease_severity_markers()
    """
    now, we add the outcome features
    """
    data_processing.add_outcome_features()
    """
    next, process drug orders
    """
    # separate the prescriptions file into several files for each of AMI-related drugs
    data_processing.write_prescriptions_files("./data/annotated/unique-medications-list-ami-patients_SPC.xls")
    # process narcotic analgesics
    process_drug_orders.process_narcotic_prescriptions()
    # process non-narcotic analgesics
    process_drug_orders.process_non_narcotic_prescriptions()
    # process aspirin orders
    process_drug_orders.process_aspirin_prescriptions()
    # process statin orders
    process_drug_orders.process_drug_prescriptions(drug_class="statin")
    # process beta-blocker orders
    process_drug_orders.process_drug_prescriptions(drug_class="beta-blocker")
    # process ace-inhibitor orders
    process_drug_orders.process_drug_prescriptions(drug_class="ace-inhibitor")
    # process non-aspirin anti-platelet orders
    process_drug_orders.process_drug_prescriptions(drug_class="anti-platelet")
    # add all drug order features to the feature file
    process_drug_orders.add_drug_features_in_patient_file()
    process_drug_orders.add_pain_therapy_type_features()
    """
    run hypothesis tests to generate p-values 
    """
    # first obtain differences for the entire-admission duration
    hypothesis_testing.analgesic_differences_entire_los()
    hypothesis_testing.aspirin_differences_entire_los()
    hypothesis_testing.procedures_differences_entire_los()
    # next, obtain differences for the first-five days of admission
    hypothesis_testing.analgesic_differences_first_five_days()
    hypothesis_testing.ami_therapies_differences_five_days()
    # finally, obtain differences in means of the doses/order counts for each of the drugs under study
    hypothesis_testing.dosage_or_order_count_differences_entire_los()
    """
    fit logistic regression models to obtain association of treatment features with outcome
    """
    outcome_analysis.main()


