import numpy as np
import csv
import random


def extract_data_suites(unprocessed):
    text = []
    labels = []
    test_text = []
    test_labels = []
    # Well shuffled data seems to perform better across runs, so x2 shuffle
    random.shuffle(unprocessed)
    random.shuffle(unprocessed)
    split_len_test = int(len(unprocessed) * 0.1)
    split_len_train = int(len(unprocessed) * 0.9)
    outlier_count_test = 0
    normal_count_test = 0
    outlier_count_train = 0
    normal_count_train = 0
    for line in unprocessed:
        text_accum = []
        try:
            text_accum.append(float(line[2]))
            text_accum.append(float(line[3]))
            text_accum.append(float(line[4]))
            text_accum.append(float(line[5]))
            text_accum.append(float(line[9]))
            text_accum.append(float(line[10]))
            text_accum.append(float(line[11]))
            if line[8].lower() == "normal":
                if normal_count_test < int(split_len_test * 0.9):
                    test_labels.append([0.0, 1.0])
                    test_text.append(text_accum)
                    normal_count_test += 1
                elif normal_count_train < int(split_len_train * 0.5):
                    labels.append([0.0, 1.0])
                    text.append(text_accum)
                    normal_count_train += 1
            elif line[8].lower() == "outlier":
                if outlier_count_test < int(split_len_test * 0.1):
                    test_labels.append([1.0, 0.0])
                    test_text.append(text_accum)
                    outlier_count_test += 1
                elif outlier_count_train < int(split_len_train * 0.5):
                    labels.append([1.0, 0.0])
                    text.append(text_accum)
                    outlier_count_train += 1
        except Exception as e:
            # Discard ambiguous data entries
            continue
    return np.array(text), np.array(labels), np.array(test_text), np.array(test_labels)


def get_data(file_name):
    unprocessed = []
    with open(file_name, 'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        flag = False
        for line in reader:
            if not flag:
                flag = True
                continue
            unprocessed.append(line)
    return extract_data_suites(unprocessed)
