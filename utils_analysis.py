from caserec.utils.process_data import ReadFile
import numpy as np
import pandas as pd

# FILE PATHS
train_file = '../../datasets/BookCrossing/folds/0/5.7/train.dat'
tl1 = '../../datasets/BookCrossing/folds/0/5.7/labeled_set_1.dat'
tl2 = '../../datasets/BookCrossing/folds/0/5.7/labeled_set_2.dat'
sep = '\t'

### Calcular quantidade de itens que foram dados notas e quantos % é do total do dataset, no train.dat ###
train_set = ReadFile(train_file, sep=sep).read()

# Dictionary with unobserved items for each user / Map user with its respective id
train_set_items_unobserved = train_set['items_unobserved']

list_percentage_items_without_ratings = list()

for user in train_set['users']:
    percentage_items_without_ratings = (1 - (len(train_set['items_seen_by_user'][user])/len(train_set['items_unobserved'][user])))*100
    list_percentage_items_without_ratings.append(percentage_items_without_ratings)

sorted_percentage_items_without_ratings = sorted(list_percentage_items_without_ratings)
print("Percentage_items_without_ratings (TRAIN.DAT). Higher: {}. Lower: {}. Mean: {}".format(sorted_percentage_items_without_ratings[-1],
                                                                                 sorted_percentage_items_without_ratings[0],
                                                                                 np.mean(list_percentage_items_without_ratings)))


### Calcular o mesmo, nos dois datasets enriquecidos
tl1_set = ReadFile(tl1, sep=sep).read()

list_percentage_items_without_ratings_tl1 = list()
for user in tl1_set['users']:
    percentage_items_without_ratings_tl1 = (1 - (len(tl1_set['items_seen_by_user'][user])/len(tl1_set['items_unobserved'][user])))*100
    list_percentage_items_without_ratings_tl1.append(percentage_items_without_ratings)
sorted_percentage_items_without_ratings_tl1 = sorted(list_percentage_items_without_ratings_tl1)
print("Percentage_items_without_ratings (TL1.DAT). Higher: {}. Lower: {}. Mean: {}".format(sorted_percentage_items_without_ratings_tl1[-1],
                                                                                 sorted_percentage_items_without_ratings_tl1[0],
                                                                                 np.mean(list_percentage_items_without_ratings_tl1)))

tl2_set = ReadFile(tl2, sep=sep).read()

list_percentage_items_without_ratings_tl2 = list()
for user in tl2_set['users']:
    percentage_items_without_ratings_tl2 = (1 - (len(tl2_set['items_seen_by_user'][user])/len(tl2_set['items_unobserved'][user])))*100
    list_percentage_items_without_ratings_tl2.append(percentage_items_without_ratings)
sorted_percentage_items_without_ratings_tl2 = sorted(list_percentage_items_without_ratings_tl2)
print("Percentage_items_without_ratings (TL2.DAT). Higher: {}. Lower: {}. Mean: {}".format(sorted_percentage_items_without_ratings_tl2[-1],
                                                                                 sorted_percentage_items_without_ratings_tl2[0],
                                                                                 np.mean(list_percentage_items_without_ratings_tl2)))


### Calcular a quantidade de zeros e uns no dataset binarizado, e a %
binary_matrix = '../../datasets/BookCrossing/folds/0/5.7/binary_matrix.dat'
binary_matrix_in = pd.read_csv(binary_matrix, sep=sep, header=None)

# Determina a quantidade de ZEROS e UNS, calcula a proporção entre eles
size_all = len(binary_matrix_in[2])
print("LEN binary_matrix_in[2]:", len(binary_matrix_in[2]))
number_zeros_in_ratings = (binary_matrix_in[2] == 0).astype(int).sum(axis=0)
number_ones_in_ratings = (binary_matrix_in[2] == 1).astype(int).sum(axis=0)
print("% ZEROS IN BINARY MATRIX", (number_zeros_in_ratings/size_all)*100)
print("% ONES IN BINARY MATRIX", (number_ones_in_ratings/size_all)*100)


## PER USER
bm_per_user = ReadFile(binary_matrix, sep=sep, as_binary=True).read()

list_percentage_items_without_ratings_bm_per_user = list()
for user in bm_per_user['users']:
    percentage_items_without_ratings_bm_per_user = (1 - (len(bm_per_user['items_seen_by_user'][user])/len(bm_per_user['items_unobserved'][user])))*100
    list_percentage_items_without_ratings_bm_per_user.append(percentage_items_without_ratings_bm_per_user)
sorted_percentage_items_without_ratings_bm_per_user = sorted(list_percentage_items_without_ratings_bm_per_user)
print("Percentage_items_without_ratings (Binary Matrix). Higher: {}. Lower: {}. Mean: {}".format(sorted_percentage_items_without_ratings_bm_per_user[-1],
                                                                                 sorted_percentage_items_without_ratings_bm_per_user[0],
                                                                                 np.mean(list_percentage_items_without_ratings_bm_per_user)))


### Calculate the sparsity of the set: N / (nu * ni)
print("train_set Sparsity:", train_set['sparsity']) 
print("tl1_set Sparsity:", tl1_set['sparsity']) 
print("tl2_set Sparsity:", tl2_set['sparsity']) 
print("Binary Matrix Sparsity:", bm_per_user['sparsity'])

