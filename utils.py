import pandas as pd


def pad_list_with_zeros(l, new_length):
    """
    Pads an array with zeros until the array achieves the desired new length
    :param l: array to be padded
    :param new_length: the new array length
    :return: array of new length padded with zeros
    """
    return (l + [0] * new_length)[:new_length]


def male_and_female_population_ami(patients_df):
    """
    Utility function to return the total number of male and female patients in a sample
    :param patients_df:
    :return:
    """
    total_males = patients_df.where(patients_df['gender'] == 'M')
    total_males = total_males[pd.notnull(total_males['HADM_ID'])]

    total_females = patients_df.where(patients_df['gender'] == 'F')
    total_females = total_females[pd.notnull(total_females['HADM_ID'])]

    return len(total_females), len(total_males)

def count_where_value_greater_than_zero(a):
    """
    Counts number of times a value greater than zero is in the list a
    :param a: A list of int/float value
    :return: A count of items in the list with a value > 0
    """
    return len([j for j in a if j > 0])


def mean(a):
    """
    compute the mean of a list
    :param a: list
    :return: mean
    """
    count = len([i for i in a if i > 0])
    try:
        return sum(a) / count
    except ZeroDivisionError:
        return 0