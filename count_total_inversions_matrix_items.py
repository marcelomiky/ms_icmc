import numpy as np
from counting_inversions import count_inversions
from distance import distance


def count_total_inversions_matrix_items(file1, file2):
    """Compare each item of file1 and file2 (output_bookcrossing_set1_items.txt and output_bookcrossing_set2_items.txt),
    and output the total of inversions"""

    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    if len(data1) != len(data2):
        print("The number of lines from both files must be the same")
        return 0

    inversions = 0
    for i in range(len(data1)):
        list1 = list(data1[i])
        list2 = list(data2[i])
        # print("i = {}, list1 = {}, list2 = {}".format(i, list1, list2))
        inversions += count_inversions(list1, list2)

    return inversions


def count_distance_matrix_items(file1, file2):
    """Compare each item of file1 and file2 (output_bookcrossing_set1_items.txt and output_bookcrossing_set2_items.txt),
    and output the total of distances"""

    data1 = np.loadtxt(file1)
    data2 = np.loadtxt(file2)

    if len(data1) != len(data2):
        print("The number of lines from both files must be the same")
        return 0

    dist = 0
    for i in range(len(data1)):
        list1 = list(data1[i])
        list2 = list(data2[i])
        dist += distance(list1, list2)

    return dist


file1 = './output_bookcrossing_set1_items.txt'
file2 = './output_bookcrossing_set2_items.txt'


inv = count_total_inversions_matrix_items(file1, file2)
print("TOTAL INVERSIONS:", inv)  # 16562
print("MEAN (TOTAL INVERSIONS):", inv / len(np.loadtxt(file1)))  # 12.78918918918919

dist = count_distance_matrix_items(file1, file2)
print("Distance:", dist)  # 26512
