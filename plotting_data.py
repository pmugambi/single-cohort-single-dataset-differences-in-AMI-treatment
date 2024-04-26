from __future__ import division
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_daily_proportions(maj_received_prop, min_received_prop, p_values, maj_label, min_label, title, save_paths,
                           max_range=5):
    """
    Given proportions of 2 groups (e.g., male vs female) that received orders for various drugs for each of
    the first five days of hospitalization, generate and save a plot of these proportions.
    :param maj_received_prop: proportion of the majority group that received the treatment
    :param min_received_prop: proportion of the minority group that received the treatment
    :param p_values: p-values of the differences in the proportions between the 2 groups, computed using an exact
    test (most likely Barnard)
    :param maj_label: the label for the plot for the majority group
    :param min_label: the label for the plot for the minority group
    :param title: the title of the plot
    :param save_paths: paths and file names to use in saving the generated images
    :param max_range: maximum number of days for which the plot should be generated (i.e., X-axis)
    :return: nothing. simply generates the images and saves them
    """

    fig, axes = plt.subplots(figsize=(9.5, 6))
    barWidth = 0.25

    r1 = np.arange(len(maj_received_prop))
    r2 = [x + barWidth for x in r1]

    # color pallete obtained from https://davidmathlogic.com/colorblind/#%23E66100-%235D3A9B.
    # Brown and blue combination, that's accessible to people with color blindness

    rects1 = axes.bar(r1, maj_received_prop, color=[(153/255, 79/255, 0)], width=barWidth,
                      edgecolor='white', label=maj_label)
    rects2 = axes.bar(r2, min_received_prop, color=[(0, 108/255, 209/255)], width=barWidth,
                      edgecolor='white', label=min_label)

    def autolabel(rects, y_val=3):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            text = height
            axes.annotate('{}'.format(text),
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, y_val),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    def add_text(x, y, text):
        for i in range(len(x)):
            axes.text(x[i], y[i], text[i], c="blue")

    max_vals = np.maximum(maj_received_prop, min_received_prop)
    text_y_ind = [i + 8 for i in max_vals]
    text_x_ind = [i for i in r1]
    pv_text_list = ["p=" + str(round(pv, 3)) for pv in list(p_values)]
    add_text(text_x_ind, text_y_ind, pv_text_list)

    axes.set_xlabel('Admit day', fontweight='bold')
    axes.set_ylabel('Count(%)', fontweight='bold')
    x_ticks = [r + barWidth for r in range(max_range + 1)]
    axes.set_xticks(x_ticks)
    x_labels = list(range(1, 1 + max_range))
    x_labels.append("all")
    axes.set_xticklabels(x_labels)
    axes.set_ylim(0, max(max_vals) + 30)
    axes.set_title(title)
    fig.tight_layout()
    plt.legend()
    for path in save_paths:
        plt.savefig(path)
    # plt.show()
    plt.close()


def plot_proportions_received_orders_daily(treatment_type, treatment_therapy, save_paths):
    """
    This function uses plot_daily_proportions() above to generate the images for all analgesia and AMI-related
    drugs under study
    :param treatment_type: whether the treatment is "analgesia" or "AMI-specific"
    :param treatment_therapy: specific drug class, e.g., narcotic, statin, aspirin
    :param save_paths: paths and file names to be used to save the generated images
    :return: nothing, plot_daily_proportions() is called, which generates and saves the images in the specified paths
    """
    if treatment_type == "analgesia":
        df = pd.read_csv("./data/results/first-five-days/p-values/differences-in-proportions-that-received-analgesia"
                         ".csv")
    elif treatment_type == "ami-drugs":
        df = pd.read_csv("./data/results/first-five-days/p-values/differences-in-proportions-that-received-ami"
                         "-therapies.csv")
    else:
        raise ValueError("expected treatment type values are; 'analgesia' or 'ami-drugs'")

    treatment_therapy_df = df[df["treatment"] == treatment_therapy]

    conditions = [None, "c-shock", "no-c-shock"]
    attributes = ["sex", "race", "age"]
    labels = [["Male", "Female"], ["White", "Non-White"], [">39", "<=39"]]
    for cond in conditions:
        if cond is None:
            cond_therapy_df = treatment_therapy_df[treatment_therapy_df["conditioned-on"].isna()]
        else:
            cond_therapy_df = treatment_therapy_df[treatment_therapy_df["conditioned-on"] == cond]

        for i in range(0, len(attributes)):
            at_cond_therapy_df = cond_therapy_df[cond_therapy_df["attribute"] == attributes[i]]
            at_cond_p_values = at_cond_therapy_df["p-value"]
            maj_received_values = at_cond_therapy_df["majority-yes"]
            maj_dn_receive_values = at_cond_therapy_df["majority-no"]
            min_received_values = at_cond_therapy_df["minority-yes"]
            min_dn_receive_values = at_cond_therapy_df["minority-no"]

            total_maj = maj_received_values + maj_dn_receive_values
            total_min = min_received_values + min_dn_receive_values

            save_name = "proportion-received-" + treatment_therapy + "-by-" + attributes[i]
            save_title = "Proportion received " + treatment_therapy + " by " + attributes[i]

            if cond == "c-shock":
                title_save = " conditioned on positive cardiogenic shock diagnosis"
                cond_save = "-conditioned-on-c-shock-positive"
            elif cond == "no-c-shock":
                title_save = " conditioned on negative cardiogenic shock diagnosis"
                cond_save = "-conditioned-on-c-shock-negative"
            else:
                title_save = ""
                cond_save = ""

            pdf_save_name = ""
            eps_save_name = ""

            for path in save_paths:
                if "pdf" in path:
                    pdf_save_name = path + save_name + cond_save + ".pdf"
                if "eps" in path:
                    eps_save_name = path + save_name + cond_save + ".eps"
            save_title += title_save

            maj_received_prop = [round(x, 1) for x in list((maj_received_values / total_maj) * 100)]
            min_received_prop = [round(x, 1) for x in list((min_received_values / total_min) * 100)]
            plot_daily_proportions(maj_received_prop=maj_received_prop, min_received_prop=min_received_prop,
                                   p_values=at_cond_p_values, maj_label=labels[i][0], min_label=labels[i][1],
                                   title=save_title,
                                   save_paths=[pdf_save_name, eps_save_name])


def main():
    """
    This iterates over all treatment types and drugs and generates all the images, by calling function
    plot_proportions_received_orders_daily() above
    :return: nothing. all generated images are saved in ./data/results/first-five-days/plots/
    """
    treatment_types = ["ami-drugs", "analgesia"]
    therapies = [["aspirin", "non-aspirin-anti-platelet", "beta-blocker", "ace-inhibitor", "statin"],
                 ["any-analgesia", "narcotics", "non-narcotics", "narcotics-only", "non-narcotics-only",
                  "combined-therapy"]]
    for i in range(len(treatment_types)):
        t_type = treatment_types[i]
        treatment_therapies = therapies[i]
        for therapy in treatment_therapies:
            pdf_path = "./data/results/first-five-days/plots/pdf/"+t_type+"/"+therapy+"/"
            eps_path = "./data/results/first-five-days/plots/eps/"+t_type+"/"+therapy+"/"
            pathlib.Path(pdf_path).mkdir(parents=True, exist_ok=True)
            pathlib.Path(eps_path).mkdir(parents=True, exist_ok=True)
            plot_proportions_received_orders_daily(treatment_type=t_type, treatment_therapy=therapy,
                                                   save_paths=[pdf_path, eps_path])


if __name__ == '__main__':
    main()




