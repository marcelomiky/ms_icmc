import numpy as np


def mean_and_std(file):
    """file to count the mean and standard deviation of the ratings matrix"""

    data = np.loadtxt(file)

    total_mean = 0
    total_std = 0

    for i in range(len(data)):
        total_mean += np.mean(data[i])
        total_std += np.std(data[i])

    total_mean /= len(data)
    total_std /= len(data)

    return total_mean, total_std


file1 = './output_bookcrossing_set1_ratings.txt'
file2 = './output_bookcrossing_set2_ratings.txt'


mean1, std1 = mean_and_std(file1)
print("MEAN of ratings of ALL MEANS (set1):", mean1)
print("MEAN OF ALL STD (set1): ", std1)

mean2, std2 = mean_and_std(file2)
print("MEAN of ratings of ALL MEANS (set2):", mean2)
print("MEAN OF ALL STD (set2): ", std2)

