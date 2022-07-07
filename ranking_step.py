from caserec.utils.process_data import ReadFile
import operator
import pandas as pd
import numpy as np

labeled_file = '../../datasets/BookCrossing/folds/0/labeled_set_1.dat'
#labeled_file = '../../datasets/BookCrossing/folds/0/labeled_set_2.dat'
#test_file ='../../datasets/BookCrossing/folds/0/test.dat'


sep = '\t'
set_to_rank = ReadFile(labeled_file, sep=sep).read()



def ranking_dict_list_users_to_top_items(set_to_rank_in):
    list_users_in = list(set_to_rank['feedback'].keys())

    all_users_items_and_ratings_dict = set_to_rank_in['feedback']

    all_users_items_and_ratings_list = []
    for i in list_users_in:
        all_users_items_and_ratings_list.append(set_to_rank_in['feedback'][i])

    first_user_items_and_ratings = all_users_items_and_ratings_list[0]
    second_user_items_and_ratings = all_users_items_and_ratings_list[1]

    matrix_ranked_items_all_users = []
    matrix_ranked_items_only_items = np.zeros(shape=(len(list_users_in), 10))  
    matrix_ranked_items_only_ratings = np.zeros(shape=(len(list_users_in), 10))  

    row = 0
    col = 0

    for usr in all_users_items_and_ratings_list:
        temp = sorted(usr.items(), key=operator.itemgetter(1), reverse=True)
        matrix_ranked_items_all_users.append(temp)
        for j in temp:
            matrix_ranked_items_only_items[row][col] = j[0]
            matrix_ranked_items_only_ratings[row][col] = j[1]
            col += 1

        row += 1
        col = 0

    return list_users_in, matrix_ranked_items_all_users, matrix_ranked_items_only_items, matrix_ranked_items_only_ratings


list_users, matrix_ranked_items_all_users, matrix_ranked_items_only_items, matrix_ranked_items_only_ratings = ranking_dict_list_users_to_top_items(set_to_rank)
