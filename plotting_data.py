from __future__ import division
import datetime

from datetime import date, timedelta

import pandas as pd
from collections import Counter
import researchpy as rp
from scipy import stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import math
from matplotlib.patches import Rectangle

import data_processing, utils
from read_data_ import count_patients_in_hospital_per_day_by_gender_and_race, \
    process_race_and_gender_patient_proportions

all_patients_df = data_processing.read_csv_file("./data/healthy_no_pmh_mi_patients_list.csv")
mi_female_totals, mi_male_totals = utils.male_and_female_population_ami(all_patients_df)


def number_prescribed_pain_relief_per_admission_day(day_dosage_df, max_range, disease):
    """
    plots a bar chart of percentage of men/women prescribed pain relief during each admit day
    :return: none
    """
    print("len of df = ", len(day_dosage_df))
    print("number of unique patients = ", len(list(day_dosage_df['HADM_ID'].unique())))
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

    men_percs = [round(x / len(men_records["HADM_ID"].unique().tolist()) * 100, 1) for x in men_daily_dosages]
    women_percs = [round((x / len(women_records["HADM_ID"].unique().tolist())) * 100, 1) for x in women_daily_dosages]

    print("men percs = ", men_percs, "women percs = ", women_percs)

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    r1 = np.arange(len(men_percs))
    r2 = [x + barWidth for x in r1]

    rects1 = axes6.bar(r1, men_percs, color='r', width=barWidth,
                       edgecolor='white', label='male')
    rects2 = axes6.bar(r2, women_percs, color='g', width=barWidth,
                       edgecolor='white', label='female')

    def autolabel(rects, y_val=3):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            rec_i = list(rects).index(rect)
            # print("rec_i = ", rec_i)
            height = rect.get_height()
            text = height
            axes6.annotate('{}'.format(text),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, y_val),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes6.text(x[i], y[i], text[i], c="blue")

    text_y_ind = [i + 8 for i in men_percs]
    # text_y_ind.append(min(text_y_ind)+8)
    # text_x_ind = [(i+j)/2 for i, j in zip(r1, r2)]
    text_x_ind = [i for i in r1]
    # adding a x axis value for the all p-value text
    # text_x_ind.append(text_x_ind[-1]+(2*barWidth))

    # add a point on the plot
    # axes6.scatter(text_x_ind[-1]+(2*barWidth), min(text_y_ind)+5, marker='.', c='white')

    print("len of x and y = ", len(text_x_ind), len(text_y_ind))

    print("text_x_ind = ", text_x_ind)
    print("text_y_ind = ", text_y_ind)

    add_text(text_x_ind, text_y_ind, ['p=0.273', 'p=0.081', 'p=0.086', 'p=0.336', 'p=0.815'])  # , 'p=0.03'])

    axes6.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range)]
    # x_ticks.append(text_x_ind[-1]+(2*barWidth))
    axes6.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    x_labels.append("all")
    axes6.set_xticklabels(x_labels)

    from matplotlib.patches import Rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    axes6.legend([rects1, rects2, extra], ("male", "female", "p-value all = 0.03"))
    # axes6.legend()
    print("max y = ", max(text_y_ind) + 10)
    plt.ylim(top=max(text_y_ind) + 10)
    # plt.tight_layout()
    axes6.set_ylabel("Percentage of patients", fontweight="bold")
    # axes6.set_title(disease+": Percentage Patients Prescribed Pain Relief each Admit Day")
    axes6.set_title("Patient percentage, by gender, prescribed pain relief each admit day")
    fig6.savefig("./results/" + disease + "_percentage_prescribed_pain_relief_per_admit_day.png", dpi=300,
                 bbox_inches='tight')
    fig6.savefig("./results/" + disease + "_percentage_prescribed_pain_relief_per_admit_day.pdf", dpi=300,
                 bbox_inches='tight')
    plt.show()


def plot_frequencies(df, column_name, disease):
    frequencies = pd.value_counts(df[column_name].values, sort=True)
    fig, ax = plt.subplots()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    x = list(frequencies.keys())
    y = list(frequencies)

    rect = ax.bar(x, y)
    ax.set_title(column_name.capitalize() + " frequencies ")
    ax.set_ylabel("Count")
    ax.set_xlabel(column_name.capitalize())
    plt.xticks(rotation=90)
    plt.tight_layout()

    autolabel(rect)
    plt.savefig("./data/visualization/new/" + disease + '_' + column_name.lower() + "_frequencies.png", dpi=300,
                bbox_inches='tight')
    plt.savefig("./data/visualization/new/" + disease + '_' + column_name.lower() + "_frequencies.pdf", dpi=300,
                bbox_inches='tight')
    plt.show()


def plot_daily_pain_dosage_days_by_gender(day_dosage_df, max_range, di):
    men_records = day_dosage_df.where(day_dosage_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = day_dosage_df.where(day_dosage_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosages = data_processing.dosage_per_admit_day(men_records, "total_dosage_days")
    men_padded_dosages = []
    for d in men_dosages:
        men_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    men_averages = list(map(data_processing.mean, zip(*men_padded_dosages)))

    women_dosages = data_processing.dosage_per_admit_day(women_records, "total_dosage_days")
    women_padded_dosages = []
    for d in women_dosages:
        women_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    women_averages = list(map(data_processing.mean, zip(*women_padded_dosages)))

    print("men daily dosages = ", men_averages)
    print("women daily dosages = ", women_averages)

    fig6, axes6 = plt.subplots()
    axes6.scatter(list(range(1, 1 + max_range)), men_averages, marker='*', c="r", label="male")
    axes6.scatter(list(range(1, 1 + max_range)), women_averages, marker='^', c="g", label="female")
    axes6.legend()
    axes6.set_xticks(list(range(1, 1 + max_range)))
    axes6.set_xticklabels(list(range(1, 1 + max_range)))
    axes6.set_ylabel("Average Dosage Days")
    axes6.set_xlabel("Admit Day")
    # axes6.set_title(di.lower()+" average dosage days per admit day")
    axes6.set_title("Average dosage days per admit day")
    # plt.tight_layout()
    fig6.savefig("./results/" + di.lower() + "_average_dosage_days_per_admit_day.png", dpi=300,
                 bbox_inches='tight')
    fig6.savefig("./results/" + di.lower() + "_average_dosage_days_per_admit_day.pdf", dpi=300,
                 bbox_inches='tight')
    plt.show()


def plot_total_dosage_days_per_gender(total_dosage_days_df, d):
    men_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosage_days = men_records["total_dosage_days"]
    women_dosage_days = women_records["total_dosage_days"]

    men_average_total_dosage_days = sum(men_dosage_days) / len(men_dosage_days)
    women_average_total_dosage_days = sum(women_dosage_days) / len(women_dosage_days)

    fig, ax = plt.subplots()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                    '%f' % round(height, 2),
                    ha='center', va='bottom')

    x = list(["Male", "Female"])
    y = list([men_average_total_dosage_days, women_average_total_dosage_days])

    rect1 = ax.bar(x[0], y[0], color='r', label="male")
    rect2 = ax.bar(x[1], y[1], color='g', label="female")
    ax.legend()
    # ax.set_title(d.lower()+" total average dosage days by gender")
    ax.set_title("Total average dosage days by gender")
    ax.set_ylabel("Average Dosage Days")
    ax.set_xlabel("Gender")
    plt.xticks(rotation=90)
    plt.tight_layout()

    autolabel(rect1)
    autolabel(rect2)
    fig.savefig("./results/" + d.lower() + "_total_average_dosage_days_by_gender.png", dpi=300,
                bbox_inches='tight')
    fig.savefig("./results/" + d.lower() + "_total_average_dosage_days_by_gender.pdf", dpi=300,
                bbox_inches='tight')
    plt.show()


def plot_daily_pain_dosage_days_by_gender_and_drug_type(day_dosage_df, max_range):
    men_records = day_dosage_df.where(day_dosage_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = day_dosage_df.where(day_dosage_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_narc_dosages = data_processing.dosage_per_admit_day(men_records, "narcotic_dosage_days")
    men_narc_padded_dosages = []
    for d in men_narc_dosages:
        men_narc_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    men_narc_averages = list(map(data_processing.mean, zip(*men_narc_padded_dosages)))

    women_narc_dosages = data_processing.dosage_per_admit_day(women_records, "narcotic_dosage_days")
    women_narc_padded_dosages = []
    for d in women_narc_dosages:
        women_narc_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    women_narc_averages = list(map(data_processing.mean, zip(*women_narc_padded_dosages)))

    men_non_narc_dosages = data_processing.dosage_per_admit_day(men_records, "non_narcotic_dosage_days")
    men_non_narc_padded_dosages = []
    for d in men_non_narc_dosages:
        men_non_narc_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    men_non_narc_averages = list(map(data_processing.mean, zip(*men_non_narc_padded_dosages)))

    women_non_narc_dosages = data_processing.dosage_per_admit_day(women_records, "non_narcotic_dosage_days")
    women_non_narc_padded_dosages = []
    for d in women_non_narc_dosages:
        women_non_narc_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    women_non_narc_averages = list(map(data_processing.mean, zip(*women_non_narc_padded_dosages)))

    print("men daily dosages = ", men_narc_averages)

    fig1, axes1 = plt.subplots()
    axes1.scatter(list(range(1, 1 + max_range)), men_narc_averages, marker='*', c="r", label="male")
    axes1.scatter(list(range(1, 1 + max_range)), women_narc_averages, marker='^', c="g", label="female")
    axes1.legend()
    axes1.set_xticks(list(range(1, 1 + max_range)))
    axes1.set_xticklabels(list(range(1, 1 + max_range)))
    axes1.set_ylabel("Average Dosage Days")
    axes1.set_xlabel("Admit Day")
    axes1.set_title("Average narcotic dosage days per admit day")
    fig1.savefig("./results/average_narcotic_dosage_days_per_admit_day.png")
    fig1.savefig("./results/average_narcotic_dosage_days_per_admit_day.pdf")

    fig2, axes2 = plt.subplots()
    axes2.scatter(list(range(1, 1 + max_range)), men_non_narc_averages, marker='o', c="r", label="male")
    axes2.scatter(list(range(1, 1 + max_range)), women_non_narc_averages, marker='x', c="g", label="female")
    axes2.legend()
    axes2.set_xticks(list(range(1, 1 + max_range)))
    axes2.set_xticklabels(list(range(1, 1 + max_range)))
    axes2.set_ylabel("Average Dosage Days")
    axes2.set_xlabel("Admit Day")
    axes2.set_title("Average non-narcotic dosage days per admit day")
    fig2.savefig("./results/average_non_narcotic_dosage_days_per_admit_day.png", dpi=300,
                 bbox_inches='tight')
    fig2.savefig("./results/average_non_narcotic_dosage_days_per_admit_day.pdf", dpi=300,
                 bbox_inches='tight')
    plt.show()


def plot_total_dosage_days_per_gender_and_drug_type(total_dosage_days_df, drug_type):
    men_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = total_dosage_days_df.where(total_dosage_days_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    men_dosage_days = men_records["total_dosage_days"]
    women_dosage_days = women_records["total_dosage_days"]

    men_average_total_dosage_days = sum(men_dosage_days) / len(men_dosage_days)
    women_average_total_dosage_days = sum(women_dosage_days) / len(women_dosage_days)
    print("checking lens == ", len(men_records), len(men_dosage_days))

    fig, ax = plt.subplots()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                    '%f' % round(height, 1),
                    ha='center', va='bottom')

    x = list(["Male", "Female"])
    y = list([men_average_total_dosage_days, women_average_total_dosage_days])

    rect1 = ax.bar(x[0], y[0], color='r', label="male")
    rect2 = ax.bar(x[1], y[1], color='g', label="female")
    ax.legend()
    ax.set_title("Total " + drug_type.lower() + " average dosage days by gender")
    ax.set_ylabel("Average Dosage Days")
    ax.set_xlabel("Gender")
    plt.xticks(rotation=90)
    plt.tight_layout()

    autolabel(rect1)
    autolabel(rect2)
    fig.savefig("./results/" + drug_type.lower() + "_total_average_dosage_days_by_gender.png", dpi=300,
                bbox_inches='tight')
    fig.savefig("./results/" + drug_type.lower() + "_total_average_dosage_days_by_gender.pdf", dpi=300,
                bbox_inches='tight')
    plt.show()


def plot_percentage_tested_for_cardiac_enzyme(summarized_lab_df, total_patients_df):
    male_tested = summarized_lab_df.where(summarized_lab_df['gender'] == 'M')
    male_tested = male_tested[pd.notnull(male_tested['HADM_ID'])]

    female_tested = summarized_lab_df.where(summarized_lab_df['gender'] == 'F')
    female_tested = female_tested[pd.notnull(female_tested['HADM_ID'])]

    total_males = total_patients_df.where(total_patients_df['gender'] == 'M')
    total_males = total_males[pd.notnull(total_males['HADM_ID'])]

    total_females = total_patients_df.where(total_patients_df['gender'] == 'F')
    total_females = total_females[pd.notnull(total_females['HADM_ID'])]

    perc_tested_male = (len(male_tested) / len(total_males)) * 100
    perc_tested_female = (len(female_tested) / len(total_females)) * 100

    fig1, ax1 = plt.subplots()

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                     '%f' % round(height, 2),
                     ha='center', va='bottom')

    x = list(["Male", "Female"])
    y = list([perc_tested_male, perc_tested_female])

    rect1 = ax1.bar(x[0], y[0], color='r', label="male")
    rect2 = ax1.bar(x[1], y[1], color='g', label="female")
    ax1.legend()
    ax1.set_title("Percentage of Patients with a Cardiac Enzyme Lab test")
    ax1.set_ylabel("Percentage of Patients")
    ax1.set_xlabel("Gender")

    autolabel(rect1)
    autolabel(rect2)
    fig1.savefig("./results/labevents_percentage_of_patients_tested_for_cardiac_enzyme.png")

    male_multi_tested = male_tested.where(male_tested["times_tested"] > 1)
    male_multi_tested = male_multi_tested[pd.notnull(male_multi_tested['HADM_ID'])]

    print("debug == ", len(male_multi_tested))

    female_multi_tested = female_tested.where(female_tested["times_tested"] > 1)
    female_multi_tested = female_multi_tested[pd.notnull(female_multi_tested['HADM_ID'])]

    perc_multi_tested_male = (len(male_multi_tested) / len(total_males)) * 100
    perc_multi_tested_female = (len(female_multi_tested) / len(total_females)) * 100

    print("debug 2 == ", perc_multi_tested_female, perc_multi_tested_male)

    fig2, ax2 = plt.subplots()

    def autolabel2(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                     '%f' % round(height, 2),
                     ha='center', va='bottom')

    x1 = list(["Male", "Female"])
    y1 = list([perc_multi_tested_male, perc_multi_tested_female])

    rect3 = ax2.bar(x1[0], y1[0], color='r', label="male")
    rect4 = ax2.bar(x1[1], y1[1], color='g', label="female")
    ax2.legend()
    ax2.set_title("Percentage of Patients with Multiple Cardiac Enzyme Lab tests")
    ax2.set_ylabel("Percentage of Patients")
    ax2.set_xlabel("Gender")

    autolabel2(rect3)
    autolabel2(rect4)
    fig2.savefig("./results/labevents_percentage_with_multiple_cardiac_enzyme_test.png")

    # refactor this: can be done by plot_frequencies()

    ages_times_tested = []
    ages = summarized_lab_df['age'].unique().tolist()
    for age in ages:
        times_tested = summarized_lab_df.where(summarized_lab_df['age'] == age)
        times_tested = times_tested[pd.notnull(times_tested['HADM_ID'])]
        tested_times = sum(times_tested['times_tested'])
        # print("age = ", age, " tested times = ", tested_times)
        ages_times_tested.append(tested_times)

    fig3, ax3 = plt.subplots()

    def autolabel3(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax3.text(rect.get_x() + rect.get_width() / 2., 1.00 * height,
                     '%d' % int(height),
                     ha='center', va='bottom')

    x2 = ages
    y2 = ages_times_tested

    rect = ax3.bar(x2, y2)
    ax3.set_title("Number of times patients of different ages were tested for Cardiac Anzyme")
    ax3.set_ylabel("Times tested")
    ax3.set_xlabel("Age")
    ax3.set_xlim(20, 51)
    autolabel3(rect)
    fig3.savefig("./results/labevents_age_vs_timestested.png")

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def number_prescribed_pain_relief_per_admission_day_by_race(day_dosage_df, max_range, disease):
    """
    plots a bar chart of percentage of men/women prescribed pain relief during each admit day
    :return: none
    """
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

    white_percs = [round(x / len(white_records["HADM_ID"].unique().tolist()) * 100, 1) for x in white_daily_dosages]
    non_white_percs = [round((x / len(non_white_records["HADM_ID"].unique().tolist())) * 100, 1) for x in
                       non_white_daily_dosages]

    print("white percs = ", white_percs, "non_white percs = ", non_white_percs)

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    r1 = np.arange(len(white_percs))
    r2 = [x + barWidth for x in r1]

    rects1 = axes6.bar(r1, white_percs, color='r', width=barWidth,
                       edgecolor='white', label='white')
    rects2 = axes6.bar(r2, non_white_percs, color='g', width=barWidth,
                       edgecolor='white', label='non_white')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axes6.annotate('{}'.format(height),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes6.text(x[i], y[i], text[i], c="blue")

    text_y_ind = [i + 8 for i in non_white_percs]
    # text_y_ind.append(min(text_y_ind)+8)
    # text_x_ind = [(i+j)/2 for i, j in zip(r1, r2)]
    text_x_ind = [i for i in r1]
    # adding a x axis value for the all p-value text
    # text_x_ind.append(text_x_ind[-1]+(2*barWidth))

    # add a point on the plot
    # axes6.scatter(text_x_ind[-1]+(2*barWidth), min(text_y_ind)+5, marker='.', c='white')

    print("len of x and y = ", len(text_x_ind), len(text_y_ind))

    print("text_x_ind = ", text_x_ind)
    print("text_y_ind = ", text_y_ind)

    add_text(text_x_ind, text_y_ind, ['p=0.07', 'p=0.312', 'p=0.976', 'p=0.246', 'p=0.018'])  # , 'p=0.003'])

    axes6.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range)]
    # x_ticks.append(text_x_ind[-1]+(2*barWidth))
    axes6.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    # x_labels.append("all")
    axes6.set_xticks(x_ticks)
    axes6.set_xticklabels(x_labels)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axes6.legend([rects1, rects2, extra], ("white", "non-white", "p-value all = 0.003"))
    # axes6.legend()
    print("max y = ", max(text_y_ind) + 10)
    plt.ylim(top=max(text_y_ind) + 10)
    axes6.set_ylabel("Percentage of patients", fontweight="bold")
    # axes6.set_title(disease+": patient percentage by race prescribed pain relief each admit day")
    axes6.set_title("Patient percentage, by race, prescribed pain relief each admit day")
    # fig6.savefig("./data/results/bar_charts/"+disease)
    fig6.savefig("./results/" + disease + "_percentage_by_race_prescribed_pain_relief_per_admit_day.png", dpi=300,
                 bbox_inches='tight')
    fig6.savefig("./results/" + disease + "_percentage_by_race_prescribed_pain_relief_per_admit_day.pdf", dpi=300,
                 bbox_inches='tight')
    plt.show()


def number_prescribed_drug_by_drug_type_and_gender(day_dosage_df, max_range, drug_type, drug_type_col, p_values,
                                                   p_value_all):
    # narcotic_dosage_days
    # print("df cols = ", list(day_dosage_df.columns))
    print("len of df = ", len(day_dosage_df))
    print("number of unique patients = ", len(list(day_dosage_df['HADM_ID'].unique())))
    men_records = day_dosage_df.where(day_dosage_df["gender"] == 'M')
    men_records = men_records[pd.notnull(men_records['HADM_ID'])]
    women_records = day_dosage_df.where(day_dosage_df["gender"] == 'F')
    women_records = women_records[pd.notnull(women_records['HADM_ID'])]

    print("total num of men = ", len(men_records["HADM_ID"].unique().tolist()))
    print("total num of women = ", len(women_records["HADM_ID"].unique().tolist()))

    men_dosages = data_processing.dosage_per_admit_day(men_records, drug_type_col)
    men_padded_dosages = []
    for d in men_dosages:
        men_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    men_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*men_padded_dosages)))

    print("men daily dosages = ", men_daily_dosages)

    women_dosages = data_processing.dosage_per_admit_day(women_records, drug_type_col)
    women_padded_dosages = []
    for d in women_dosages:
        women_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    women_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*women_padded_dosages)))
    print("women daily dosages = ", women_daily_dosages)

    men_percs = [round(x / len(men_records["HADM_ID"].unique().tolist()) * 100, 1) for x in men_daily_dosages]
    women_percs = [round((x / len(women_records["HADM_ID"].unique().tolist())) * 100, 1) for x in women_daily_dosages]

    print("men percs = ", men_percs, "women percs = ", women_percs)

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    r1 = np.arange(len(men_percs))
    r2 = [x + barWidth for x in r1]

    rects1 = axes6.bar(r1, men_percs, color='r', width=barWidth,
                       edgecolor='white', label='male')
    rects2 = axes6.bar(r2, women_percs, color='g', width=barWidth,
                       edgecolor='white', label='female')

    def autolabel(rects, y_val=3):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            rec_i = list(rects).index(rect)
            # print("rec_i = ", rec_i)
            height = rect.get_height()
            text = height
            axes6.annotate('{}'.format(text),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, y_val),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes6.text(x[i], y[i], text[i], c="blue")

    text_y_ind = [max(i) + 8 for i in list(zip(men_percs, women_percs))]
    text_x_ind = [i for i in r1]
    # add_text(text_x_ind, text_y_ind, ['p=0.273', 'p=0.081', 'p=0.086', 'p=0.336', 'p=0.815'])  # , 'p=0.03'])
    add_text(text_x_ind, text_y_ind, p_values)

    axes6.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range)]
    axes6.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    x_labels.append("all")
    axes6.set_xticklabels(x_labels)

    from matplotlib.patches import Rectangle
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

    axes6.legend([rects1, rects2, extra], ("male", "female", "p-value all = " + p_value_all))
    print("max y = ", max(text_y_ind) + 10)
    plt.ylim(top=max(text_y_ind) + 10)
    axes6.set_ylabel("Percentage of patients", fontweight="bold")
    # axes6.set_title(disease+": Percentage Patients Prescribed Pain Relief each Admit Day")
    axes6.set_title("Patient percentage, by gender, prescribed " + drug_type + " each admit day")
    fig6.savefig("./results/" + drug_type + "_percentage_prescribed_" + drug_type + "_per_admit_day.png", dpi=300,
                 bbox_inches='tight')
    fig6.savefig("./results/" + drug_type + "_percentage_prescribed_" + drug_type + "_per_admit_day.pdf", dpi=300,
                 bbox_inches='tight')
    plt.show()


def number_prescribed_drug_by_drug_type_and_race(day_dosage_df, max_range, drug_type, drug_type_col, p_values,
                                                 p_value_all):
    """
    plots a bar chart of percentage of men/women prescribed pain relief during each admit day
    :return: none
    """
    white = ["WHITE"]
    white_records = day_dosage_df[day_dosage_df.ETHNICITY.isin(white)]

    white_and_unspecified = ["WHITE", "UNKNOWN/NOT SPECIFIED"]
    non_white_records = day_dosage_df[~day_dosage_df.ETHNICITY.isin(white_and_unspecified)]

    print("len of white_records = ", len(white_records["HADM_ID"].unique()))
    # print(white_records.head())

    print("len of non_white_records  = ", len(non_white_records["HADM_ID"].unique()))
    # print(non_white_records.head())

    white_dosages = data_processing.dosage_per_admit_day(white_records, drug_type_col)
    white_padded_dosages = []
    for d in white_dosages:
        white_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    white_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero, zip(*white_padded_dosages)))

    print("white daily dosages = ", white_daily_dosages)

    non_white_dosages = data_processing.dosage_per_admit_day(non_white_records, drug_type_col)
    non_white_padded_dosages = []
    for d in non_white_dosages:
        non_white_padded_dosages.append(data_processing.pad_list_with_zeros(d, max_range))
    non_white_daily_dosages = list(map(data_processing.count_where_value_greater_than_zero,
                                       zip(*non_white_padded_dosages)))
    print("non_white daily dosages = ", non_white_daily_dosages)

    white_percs = [round(x / len(white_records["HADM_ID"].unique().tolist()) * 100, 1) for x in white_daily_dosages]
    non_white_percs = [round((x / len(non_white_records["HADM_ID"].unique().tolist())) * 100, 1) for x in
                       non_white_daily_dosages]

    print("white percs = ", white_percs, "non_white percs = ", non_white_percs)

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    r1 = np.arange(len(white_percs))
    r2 = [x + barWidth for x in r1]

    rects1 = axes6.bar(r1, white_percs, color='r', width=barWidth,
                       edgecolor='white', label='white')
    rects2 = axes6.bar(r2, non_white_percs, color='g', width=barWidth,
                       edgecolor='white', label='non_white')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            axes6.annotate('{}'.format(height),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes6.text(x[i], y[i], text[i], c="blue")

    text_y_ind = [max(i) + 8 for i in list(zip(white_percs, non_white_percs))]
    text_x_ind = [i for i in r1]
    add_text(text_x_ind, text_y_ind, p_values)

    axes6.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range)]
    axes6.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    axes6.set_xticks(x_ticks)
    axes6.set_xticklabels(x_labels)

    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    axes6.legend([rects1, rects2, extra], ("white", "non-white", "p-value all = " + p_value_all))
    print("max y = ", max(text_y_ind) + 10)
    plt.ylim(top=max(text_y_ind) + 10)
    axes6.set_ylabel("Percentage of patients", fontweight="bold")
    axes6.set_title("Patient percentage, by race, prescribed " + drug_type + " each admit day")
    # fig6.savefig("./data/results/bar_charts/"+disease)
    fig6.savefig("./results/" + drug_type + "_percentage_by_race_prescribed_" + drug_type + "_per_admit_day.png",
                 dpi=300, bbox_inches='tight')
    fig6.savefig("./results/" + drug_type + "_percentage_by_race_prescribed_" + drug_type + "_per_admit_day.pdf",
                 dpi=300, bbox_inches='tight')
    plt.show()


def plot_daily_prescription_proportions(filename, maj_r_cname, min_r_cname, maj_dnr_cname, min_dnr_cname,
                                        maj_label, min_label, title, save_path):
    df = pd.read_csv(filename)
    p_values = df["p-value"]
    maj_received = df[maj_r_cname]
    maj_dn_receive = df[maj_dnr_cname]
    min_received = df[min_r_cname]
    min_dn_receive = df[min_dnr_cname]

    total_maj = maj_received + maj_dn_receive
    total_min = min_received + min_dn_receive

    # maj_received_prop = (float(maj_received) / float(total_maj)) * 100
    maj_received_prop = list((maj_received / total_maj) * 100)
    maj_received_prop = [round(x, 1) for x in maj_received_prop]
    # min_received_prop = (float(min_received) / float(total_min)) * 100
    min_received_prop = list((min_received / total_min) * 100)
    min_received_prop = [round(x, 1) for x in min_received_prop]

    print("maj received prop = ", maj_received_prop)
    print("min received prop = ", min_received_prop)

    fig6, axes6 = plt.subplots()
    barWidth = 0.25

    r1 = np.arange(len(maj_received_prop))
    r2 = [x + barWidth for x in r1]

    rects1 = axes6.bar(r1, maj_received_prop, color='r', width=barWidth,
                       edgecolor='white', label=maj_label)
    rects2 = axes6.bar(r2, min_received_prop, color='g', width=barWidth,
                       edgecolor='white', label=min_label)

    def autolabel(rects, y_val=3):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            rec_i = list(rects).index(rect)
            # print("rec_i = ", rec_i)
            height = rect.get_height()
            text = height
            axes6.annotate('{}'.format(text),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, y_val),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes6.text(x[i], y[i], text[i], c="blue")

    max_vals = np.maximum(maj_received_prop, min_received_prop)
    print("max_vals = ", max_vals)

    text_y_ind = [i + 8 for i in max_vals]
    # text_y_ind.append(min(text_y_ind)+8)
    # text_x_ind = [(i+j)/2 for i, j in zip(r1, r2)]
    text_x_ind = [i for i in r1]
    # adding a x axis value for the all p-value text
    # text_x_ind.append(text_x_ind[-1]+(2*barWidth))

    # add a point on the plot
    # axes6.scatter(text_x_ind[-1]+(2*barWidth), min(text_y_ind)+5, marker='.', c='white')

    print("len of x and y = ", len(text_x_ind), len(text_y_ind))

    print("text_x_ind = ", text_x_ind)
    print("text_y_ind = ", text_y_ind)

    # TODO: stop hard coding this bit
    pv_text_list = ["p=" + str(round(pv, 3)) for pv in list(p_values)]
    print("pv_test_list = ", pv_text_list)
    # add_text(text_x_ind, text_y_ind, ['p=0.273', 'p=0.081', 'p=0.086', 'p=0.336', 'p=0.815', 'p=0.03'])
    add_text(text_x_ind, text_y_ind, pv_text_list)

    # TODO: may need to make this an input to the function
    max_range = 5

    axes6.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range + 1)]
    # x_ticks.append(text_x_ind[-1]+(2*barWidth))
    axes6.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    x_labels.append("all")
    axes6.set_xticklabels(x_labels)
    axes6.set_ylim(0, max(max_vals) + 30)
    axes6.set_title(title)
    fig6.tight_layout()
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def plot_gender_and_race_daily_proportions(keys, title, save_path):
    patients_in_hosp = count_patients_in_hospital_per_day_by_gender_and_race()
    white_males = []
    non_white_males = []
    white_females = []
    non_white_females = []
    for i in range(0, 5):
        day_total_patient_counts = patients_in_hosp[i]
        # print("day_total_patient_counts =  ", day_total_patient_counts)
        received_treatment_on_day_i = process_race_and_gender_patient_proportions(df_key=keys[i])
        # print("received_treatment_on_day_i = ", received_treatment_on_day_i)
        percentages = [round((received_treatment_on_day_i[j] / day_total_patient_counts[j]) * 100, 1)
                       for j in range(len(day_total_patient_counts))]
        # print("i = ", i, "percentages = ", percentages)
        white_males.append(percentages[0])
        non_white_males.append(percentages[1])
        white_females.append(percentages[2])
        non_white_females.append(percentages[3])

    fig, axes = plt.subplots()
    barWidth = 0.25

    r0 = np.arange(len(white_males))
    r1 = [x - (barWidth * 2) for x in r0]
    r2 = [x - barWidth for x in r0]
    r3 = [x for x in r0]
    r4 = [x + barWidth for x in r0]
    # r5 = [x + (barWidth * 2) for x in r0]

    rects1 = axes.bar(r1, white_males, color='r', width=barWidth,
                      edgecolor='white', label="white_male")
    rects2 = axes.bar(r2, non_white_males, color='g', width=barWidth,
                      edgecolor='white', label="non_white_male")
    rects3 = axes.bar(r3, white_females, color='b', width=barWidth,
                      edgecolor='white', label="white_female")
    rects4 = axes.bar(r4, non_white_females, color='c', width=barWidth,
                      edgecolor='white', label="non_white_female")

    # rects5 = axes.bar(r5, [100, 100, 100, 100, 100], color='white',
    #                   width=barWidth, edgecolor='white', label="")

    def autolabel(rects, y_val=1):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            rec_i = list(rects).index(rect)
            # print("rec_i = ", rec_i)
            height = rect.get_height()
            text = height
            axes.annotate('{}'.format(text),
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, y_val),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom', rotation=45)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    # TODO: may need to make this an input to the function
    max_range = 5
    axes.set_xlabel('Admit day', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range)]
    print("x ticks = ", x_ticks)
    axes.set_xticks(x_ticks)
    x_labels = list(range(1, max_range + 1))
    axes.set_xticklabels(x_labels)
    axes.set_ylim(0, 120)
    axes.set_title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# data = data_processing.read_csv_file("./test_file.csv")
# data = data_processing.read_csv_file("./data/heart_attack_sample_original.csv")
# data = data_processing.read_csv_file("./data/original-results-pres-patients-pd=pneumonia-maxage=50.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_pain_prescribs-with-aspirin.csv")
# print("mi patients count = ", len(data_processing.read_column_values(data, "HADM_ID").unique().tolist()))
# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_aspirin_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_stroke_prescribs.csv")
# data = data_processing.read_csv_file("./data/leq_65_no_pmh_stroke_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_diabetes_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_no_pmh_hypertension_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_essential_and_heart_ht_no_pmh_pain_prescribs.csv")
# data = data_processing.read_csv_file("./data/healthy_pneumonia_and_flue_no_pmh_pain_prescribs.csv")

# df = data_processing.read_csv_file("./data/healthy_no_pmh_mi_patients_list.csv")
# plot_frequencies(df, "gender", "mi")
# plot_frequencies(df, "INSURANCE", "mi")
# plot_frequencies(df, "ETHNICITY", "mi")
# plot_frequencies(df, "age", "mi")
# plot_frequencies(df, "DISCHARGE_LOCATION", "mi")

# df = data_processing.assign_is_narcotic_value(data)
# df = data_processing.assign_dosage_days(df)
# print("min value of dosage days = ", min(df['dosage_days'].tolist()))
# gf = data_processing.compute_dosage_per_day(df)
# print(gf.head())
# print("len of gf = ", len(gf))
# hf = data_processing.sum_dosage_per_day(gf)
# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# number_prescribed_pain_relief_per_admission_day(hf, 5, "MI")
# number_prescribed_pain_relief_per_admission_day_by_race(hf, 5, "MI")

# number_prescribed_drug_by_drug_type_and_gender(hf, 5, "narcotic", "narcotic_dosage_days",
#                                                p_values=["p=0.836", "p=0.912", "p=0.571", "p=0.503", "p=0.755"],
#                                                p_value_all="0.407")
# number_prescribed_drug_by_drug_type_and_gender(hf, 5, "non-narcotic", "non_narcotic_dosage_days",
#                                                p_values=["p=0.237", "p=0.102", "p=0.308", "p=0.574", "p=0.83"],
#                                                p_value_all="0.0501")
#

# number_prescribed_drug_by_drug_type_and_race(hf, 5, "narcotic", "narcotic_dosage_days",
#                                              p_values=["p=0.170", "p=0.517", "p=0.571", "p=0.631", "p=0.597"],
#                                              p_value_all="0.264")

# number_prescribed_drug_by_drug_type_and_race(hf, 5, "non-narcotic", "non_narcotic_dosage_days",
#                                              p_values=["p=0.027", "p=0.076", "p=0.774", "p=0.251", "p=0.007"],
#                                              p_value_all="0.00031")

# plot_daily_pain_dosage_days_by_gender(hf, 10, "MI")
#
# plot_total_dosage_days_per_gender(mf, "MI")
#
# plot_daily_pain_dosage_days_by_gender_and_drug_type(hf, 20)
#
# mmf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "narcotic_dosage_days")
# mmmf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "non_narcotic_dosage_days")
# #
# plot_total_dosage_days_per_gender_and_drug_type(mmf, "Narcotic")
# plot_total_dosage_days_per_gender_and_drug_type(mmmf, "Non-narcotic")
#
# lab_data = data_processing.read_csv_file("./data/healthy_no_pmh_cardiac_enzyme_lab_test.csv")
# lab_df = data_processing.process_lab_test(lab_data)

# patients_data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_patients_list.csv")
# plot_percentage_tested_for_cardiac_enzyme(lab_df, patients_data)

# data = data_processing.read_csv_file("./data/healthy_no_pmh_mi_aspirin_prescribs.csv")
# # print("aspirin patients count = ", len(data_processing.read_column_values(data, "HADM_ID").unique().tolist()))
# df = data_processing.assign_is_narcotic_value(data)
# df = data_processing.assign_dosage_days(df)
# gf = data_processing.compute_dosage_per_day(df)
# hf = data_processing.sum_dosage_per_day(gf)
# mf = data_processing.obtain_total_dosage_days_per_adm_id(hf, "total_dosage_days")
# number_prescribed_pain_relief_per_admission_day(hf, 5, "Aspirin")
# plot_daily_pain_dosage_days_by_gender(hf, 10, "Aspirin")
# plot_total_dosage_days_per_gender(mf, "Aspirin")
#
# number_prescribed_pain_relief_per_admission_day_by_race(hf, 5, "Aspirin")

# General behaviour
# 1. Received any pain medication
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-pain-meds-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received any pain medication by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-pain-medication.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-pain-medication.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-pain-meds-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received any pain medication by race",
    # save_path="./data/analysis-v2/h-tests/race-received-pain-medication.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-pain-medication.pdf")

# 2. Received any narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received any narcotics by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-narcotics.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-narcotics.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received any narcotics by race",
    # save_path="./data/analysis-v2/h-tests/race-received-narcotics.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-narcotics.pdf")

# 3. Received any non-narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-non-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received any non-narcotics by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-non-narcotics.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-non-narcotics.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-non-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received any non-narcotics by race",
    # save_path="./data/analysis-v2/h-tests/race-received-non-narcotics.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-non-narcotics.pdf")

# 4. Received narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-narcs-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received narcotics only by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-narcotics-only.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received narcotics only by race",
    # save_path="./data/analysis-v2/h-tests/race-received-narcotics-only.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-narcotics-only.pdf")

# 5. Received non-narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-non-narcs-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received non-narcotics only by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-non-narcotics-only.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-non-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received non-narcotics only by race",
    # save_path="./data/analysis-v2/h-tests/race-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-non-narcotics-only.pdf")

# 6. Received combined treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/gender-received-combined-pain-treatment-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received combined treatment by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-combined-pain-treatment.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-combined-pain-treatment.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/v2/race-received-combined-pain-treatment-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received combined treatment only by race",
    # save_path="./data/analysis-v2/h-tests/race-received-combined-pain-treatment.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-combined-pain-treatment.pdf")

# 6. Received aspirin treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/gender-received-aspirin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received Aspirin by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-aspirin.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-aspirin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/race-received-aspirin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received Aspirin by race",
    # save_path="./data/analysis-v2/h-tests/race-received-aspirin.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-aspirin.pdf")

# 7. Received statin treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/gender-received-statin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received statins by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-statin.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-statin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/race-received-statin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received statins by race",
    # save_path="./data/analysis-v2/h-tests/race-received-statin.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-statin.pdf")

# 8. Received anti-platelet treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/gender-received-anti-platelet-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received anti-platelets by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-anti-platelet.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-anti-platelet.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/race-received-anti-platelet-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received anti-platelets by race",
    # save_path="./data/analysis-v2/h-tests/race-received-anti-platelet.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-anti-platelet.pdf")

# 9. Received ace-inhibitor treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/gender-received-ace-inhibitor-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received ace-inhibitors by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-ace-inhibitor.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-ace-inhibitor.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/race-received-ace-inhibitor-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received ace-inhibitors by race",
    # save_path="./data/analysis-v2/h-tests/race-received-ace-inhibitor.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-ace-inhibitor.pdf")

# 10. Received beta-blockers treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/gender-received-beta-blocker-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Proportion received beta-blockers by sex",
    # save_path="./data/analysis-v2/h-tests/gender-received-beta-blockers.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/gender-received-beta-blockers.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/h-tests/race-received-beta-blocker-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Proportion received beta-blockers only by race",
    # save_path="./data/analysis-v2/h-tests/race-received-beta-blockers.png")
    save_path="./data/analysis-v2/h-tests/v2/pdfs/race-received-beta-blockers.pdf")

# With shock diagnoses analyses
# 1. Received any pain medication
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-pain-meds-p-values"
             ".csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received any pain medication by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-pain-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-pain-medication.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received-pain"
             "-meds-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received any pain medication by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-pain-medication"
    #           ".png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-pain-medication"
              ".pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-pain-meds-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received any pain medication by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-pain-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-pain-medication.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received-pain"
             "-meds-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received any pain medication by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-pain-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-pain-medication.pdf")

# 2. Received any narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received any narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received-narcs"
             "-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received any narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-narcotics.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received any narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received-narcs-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received any narcotics by race",
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-narcotics.png")
# save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-narcotics.pdf")

# 3. Received any non-narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-non-narcs-p-values"
             ".csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received any non-narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-non-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received-non"
             "-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received any non-narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-non-narcotics.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-non-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received any non-narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-non-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received-non"
             "-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received any non-narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-non-narcotics.pdf")

# 4. Received narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-narcs-only-p-values"
             ".csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received-narcs"
             "-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-narcotics-only"
    #           ".png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-narcotics-only"
              ".pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received-narcs"
             "-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-narcotics-only.pdf")

# 5. Received non-narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-non-narcs-only-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received non-narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-non-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received-non"
             "-narcs-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received non-narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-non-narcotics"
    #           "-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-non-narcotics"
              "-only.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-non-narcs-only-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received non-narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-non-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received-non"
             "-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received non-narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-non-narcotics"
    #           "-only.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-non-narcotics"
              "-only.pdf")

# 6. Received combined treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-gender-received-combined-pain"
             "-treatment-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received combined treatment by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-combined-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-combined-treatment.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-gender-received"
             "-combined-pain-treatment-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received combined treatment by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-combined"
    #           "-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-combined"
              "-treatment.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-shock/-shock-race-received-combined-pain"
             "-treatment-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received any combined treatment by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-combined-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-combined-treatment.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/with-cardiogenic-shock/-c-shock-race-received"
             "-combined-pain-treatment-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received combined treatment by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-combined"
    #           "-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-combined"
              "-treatment.pdf")

# 7. Received aspirin
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-gender-received-aspirin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received aspirin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-aspirin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-gender-received"
             "-aspirin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received aspirin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-aspirin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-race-received-aspirin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received aspirin by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-aspirin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-race-received-aspirin"
             "-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received aspirin by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-aspirin.pdf")

# 8. Received statin
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-gender-received-statin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received statin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-statin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-gender-received"
             "-statin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received statin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-statin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-race-received-statin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received statin by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-statin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-race-received-statin"
             "-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received statin by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-statin.pdf")

# 9. Received ace-inhibitors
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-gender-received-ace-inhibitor-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received ace-inhibitors by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-ace-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-ace-inhibitors.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-gender-received"
             "-ace-inhibitor-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received ace-inhibitors by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-ace-inhibitors"
    #           ".png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-ace-inhibitors"
              ".pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-race-received-ace-inhibitor-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received ace-inhibitors by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-aspirin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-race-received-ace"
             "-inhibitor-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received ace-inhibitors by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-ace-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-ace-inhibitors.pdf")

# 10. Received beta-blockers
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-gender-received-beta-blocker-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received beta-blockers by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-beta-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-beta-blockers.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-gender-received"
             "-beta-blocker-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received beta-blockers by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-beta-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-beta-blockers.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-race-received-beta-blocker-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received beta-blockers by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-beta-blocker.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-beta-blocker.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-race-received-beta"
             "-blocker-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received beta-blockers by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-beta-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-beta-blockers.pdf")

# 11. Received anti-platelets
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-gender-received-anti-platelet-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Shock \n Proportion received anti-platelets by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-anti-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-gender-received-anti-platelets.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-gender-received"
             "-anti-platelet-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Cardiogenic shock \n Proportion received anti-platelets by sex",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-anti-platelets"
    #           ".png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-gender-received-anti-platelets"
              ".pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-shock/shock-race-received-anti-platelet-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Shock \n Proportion received anti-platelets by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-anti-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/with-shock/shock-race-received-anti-platelets.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/with-cardiogenic-shock/c-shock-race-received-anti"
             "-platelet-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Cardiogenic shock \n Proportion received anti-platelets by race",
    # save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-anti-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/with-cardiogenic-shock/c-shock-race-received-anti-platelets.pdf")

# Without shock diagnoses analyses
# 1. Received any pain medication
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-pain-meds-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received any pain medication by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-pain-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-pain-medication.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-pain-meds-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received any pain medication by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-pain"
    #           "-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-pain"
              "-medication.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-pain-meds-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received any pain medication by race",
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-pain-medication.png")
# save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-pain-medication.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-pain-meds-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received any pain medication by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-pain"
    #           "-medication.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-pain"
              "-medication.pdf")

# 2. Received any narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-narcs-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received any narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received any narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-narcotics"
    #           ".png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-narcotics"
              ".pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-narcs-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received any narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received any narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-narcotics.pdf")

# 3. Received any non-narcotics
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-non-narcs-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received any non-narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-non-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-non-narcs-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received any non-narcotics by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-non"
    #           "-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-non"
              "-narcotics.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-non-narcs-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received any non-narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-non-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-non-narcotics.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-non-narcs-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received any non-narcotics by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-non"
    #           "-narcotics.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-non"
              "-narcotics.pdf")

# 4. Received narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-narcs-only-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-narcs-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-narcotics"
    #           "-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-narcotics"
              "-only.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-narcs-only-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-narcotics"
    #           "-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-narcotics"
              "-only.pdf")

# 5. Received non-narcotics only
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-non-narcs"
             "-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received non-narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-non-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-non-narcs-only-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received non-narcotics-only by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-non"
    #           "-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-non"
              "-narcotics-only.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-non-narcs-only"
             "-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received non-narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-non-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-non-narcotics-only.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-non-narcs-only-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received non-narcotics-only by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-non"
    #           "-narcotics-only.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-non"
              "-narcotics-only.pdf")

# 6. Received combined treatment
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-gender-received-combined-pain"
             "-treatment-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received combined treatment by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-combined-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-combined-treatment.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-gender-received"
             "-combined-pain-treatment-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received combined treatment by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-combined"
    #           "-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-combined"
              "-treatment.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-shock/-wo-shock-race-received-combined-pain"
             "-treatment-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received any combined treatment by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-combined-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-combined-treatment.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/pain-management/without-cardiogenic-shock/-wo-c-shock-race-received"
             "-combined-pain-treatment-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received combined treatment by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-combined"
    #           "-treatment.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-combined"
              "-treatment.pdf")

# 7. Received aspirin
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-gender-received-aspirin-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received aspirin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-aspirin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-gender-received"
             "-aspirin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received aspirin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-aspirin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-race-received-aspirin-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received aspirin by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-aspirin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-race-received"
             "-aspirin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received aspirin by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-aspirin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-aspirin.pdf")

# 8. Received statin
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-gender-received-statin-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received statin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-statin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-gender-received"
             "-statin-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received statin by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-statin.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-race-received-statin-p-values"
             ".csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received statin by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-statin.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-race-received"
             "-statin-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received statin by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-statin.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-statin.pdf")

# 9. Received ace-inhibitors
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-gender-received-ace-inhibitor-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received ace-inhibitors by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-ace-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-ace-inhibitors.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-gender-received"
             "-ace-inhibitor-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received ace-inhibitors by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-ace"
    #           "-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-ace"
              "-inhibitors.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-race-received-ace-inhibitor-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received ace-inhibitors by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-ace-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-ace-inhibitors.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-race-received"
             "-ace-inhibitor-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received ace-inhibitors by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-ace"
    #           "-inhibitors.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-ace"
              "-inhibitors.pdf")

# 10. Received beta-blockers
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-gender-received-beta-blocker-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received beta-blockers by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-beta-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-beta-blockers.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-gender-received"
             "-beta-blocker-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received beta-blockers by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-beta"
    #           "-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-beta"
              "-blockers.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-race-received-beta-blocker-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received beta-blockers by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-beta-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-beta-blockers.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-race-received"
             "-beta-blocker-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received beta-blockers by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-beta"
    #           "-blockers.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-beta"
              "-blockers.pdf")

# 11. Received anti-platelets
# a. gender
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-gender-received-anti-platelet-p"
             "-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without shock \n Proportion received anti-platelets by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-anti-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-gender-received-anti-platelets.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-gender-received"
             "-anti-platelet-p-values.csv",
    maj_r_cname="m_r", maj_dnr_cname="m_nr", min_r_cname="f_r", min_dnr_cname="f_nr",
    maj_label="male", min_label="female",
    title="Without cardiogenic shock \n Proportion received anti-platelets by sex",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-anti"
    #           "-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-gender-received-anti"
              "-platelets.pdf")

# b. race
plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-shock/wo-shock-race-received-anti-platelet-p"
             "-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without shock \n Proportion received anti-platelets by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-anti-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/without-shock/wo-shock-race-received-anti-platelets.pdf")

plot_daily_prescription_proportions(
    filename="./data/analysis-v2/shock/p-values/ami-medications/without-cardiogenic-shock/wo-c-shock-race-received"
             "-anti-platelet-p-values.csv",
    maj_r_cname="w_r", maj_dnr_cname="w_nr", min_r_cname="nw_r", min_dnr_cname="nw_nr",
    maj_label="white", min_label="non-white",
    title="Without cardiogenic shock \n Proportion received anti-platelets by race",
    # save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-anti"
    #           "-platelets.png")
    save_path="./data/analysis-v2/shock/visualizations/without-cardiogenic-shock/wo-c-shock-race-received-anti"
              "-platelets.pdf")

# TODO: rerun to write files into the correct folders


# narc_keys = ["received_narc_day1?", "received_narc_day2?", "received_narc_day3?", "received_narc_day4?",
#              "received_narc_day5?"]
# non_narc_keys = ["received_n_narc_day1?", "received_n_narc_day2?", "received_n_narc_day3?", "received_n_narc_day4?",
#                  "received_n_narc_day5?"]
# narc_only_keys = []
# plot_gender_and_race_daily_proportions(keys=narc_keys,
#                                        title="Proportions received any narcotics by gender and race",
#                                        # save_path="./data/analysis-v2/visualizations/gar-received-any-narcotics.png")
#                                        save_path="./data/analysis-v2/visualizations/gar-received-any-narcotics.pdf")
# plot_gender_and_race_daily_proportions(keys=non_narc_keys,
#                                        title="Proportions received any non-narcotics by gender and race",
#                                        # save_path="./data/analysis-v2/visualizations/gar-received-any-non-narcotics.png")
#                                        save_path="./data/analysis-v2/visualizations/gar-received-any-non-narcotics.pdf")
