import numpy as np
import csv
import random


RANDOM_SEED = 1000


def process_line_values(line, text_accum, max_val_tracker):
    tup = \
        float(line[2]), float(line[3]), float(line[4]), \
        float(line[5]), float(line[9]), float(line[10]), \
        float(line[11])
    # These values are the relevant numeric data from row
    val1, val2, val3, val4, val5, val6, val7 = tup
    if val1 > max_val_tracker[0]:
        max_val_tracker[0] = val1
    if val2 > max_val_tracker[1]:
        max_val_tracker[1] = val2
    if val3 > max_val_tracker[2]:
        max_val_tracker[2] = val3
    if val4 > max_val_tracker[3]:
        max_val_tracker[3] = val4
    if val5 > max_val_tracker[4]:
        max_val_tracker[4] = val5
    if val6 > max_val_tracker[5]:
        max_val_tracker[5] = val6
    if val7 > max_val_tracker[6]:
        max_val_tracker[6] = val7
    text_accum.extend(tup)
    data_label = line[8].lower()
    return data_label


def extract_data_suites(unprocessed):
    global RANDOM_SEED
    text = []
    labels = []
    test_text = []
    test_labels = []
    split_len_test = int(len(unprocessed) * 0.15)
    split_len_train = int(len(unprocessed) * 0.85)
    outlier_count_test = 0
    normal_count_test = 0
    outlier_count_train = 0
    normal_count_train = 0
    max_val_tracker = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Well shuffled data seems to perform better across runs, so x2 shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(unprocessed)
    random.shuffle(unprocessed)
    for line in unprocessed:
        text_accum = []
        try:
            data_label = process_line_values(line, text_accum, max_val_tracker)
            if data_label == "normal":
                if normal_count_test < int(split_len_test * 0.5):
                    test_labels.append([0.0, 1.0])
                    test_text.append(text_accum)
                    normal_count_test += 1
                elif normal_count_train < int(split_len_train * 0.5):
                    labels.append([0.0, 1.0])
                    text.append(text_accum)
                    normal_count_train += 1
            elif data_label == "outlier":
                if outlier_count_test < int(split_len_test * 0.5):
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
    # Normalize data
    for row in text:
        for i in range(len(row)):
            row[i] /= max_val_tracker[i]
    for row in test_text:
        for i in range(len(row)):
            row[i] /= max_val_tracker[i]
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
