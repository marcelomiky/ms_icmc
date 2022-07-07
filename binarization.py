from caserec.utils.process_data import ReadFile
import numpy as np
import pandas as pd
from caserec.utils.extra_functions import ComputeBui


# Find out if exists that rating for an user and an item
def find_in_dict(user, item, dict_):
    if item in dict_['users_viewed_item'].keys() and user in dict_['users_viewed_item'][item]:
        return 1
    else:
        return 0



def get_user_and_trainset_return_mean_ratings(train_set, user):
    lista_da_vez = list(train_set['feedback'][user].values())
    mean_lista_da_vez = np.mean(lista_da_vez)
    return mean_lista_da_vez


# Métrica de confiança copiado do CoRec
def pc(bui, label_data, user, item, feedback):

    nu = len(label_data['items_seen_by_user'].get(user, []))
    ni = len(label_data['users_viewed_item'].get(item, []))

    # compute bui and error
    try:
        den = np.fabs(bui[user][item] - feedback)
    except KeyError:
        den = np.fabs(label_data['mean_value'] - feedback)

    if den == 0:
        den = 0.001

    # compute confidence
    c = (nu * ni) * (1 / den)

    return c



# Binarization - train_set, tl1, tl2
def single_matrix_and_binarization(train_file, tl1, tl2, sep):

    train_set = ReadFile(train_file, sep=sep).read()
    tl1_in = pd.read_csv(tl1, sep=sep, header=None, encoding="utf-8-sig")
    tl2_in = pd.read_csv(tl2, sep=sep, header=None, encoding="utf-8-sig")

    df_to_concat = pd.DataFrame()
    df_binario = pd.DataFrame()
    listinha = []
    listinha_do_binario = []

    cycle = 1
    size = len(tl1_in)
    bui = ComputeBui(train_set).execute()

    for row, itm in tl1_in.iterrows():

        rating_tl2 = tl2_in.loc[(tl2_in[0] == itm[0]) & (tl2_in[1] == itm[1])][2]  # for that user and that item
        mean_rating = (rating_tl2 + itm[2]) / 2

        rating_tl2 = float(rating_tl2)

        list_temp = [int(itm[0]), int(itm[1]), float(mean_rating)]
        listinha.append(list_temp)

        c_tl1 = pc(bui, train_set, itm[0], itm[1], itm[2])
        c_tl2 = pc(bui, train_set, itm[0], itm[1], rating_tl2)

        norm = ((itm[2] * c_tl1 + rating_tl2 * c_tl2) / (c_tl1 + c_tl2))
        norm = float(norm)

        global_mean_user = get_user_and_trainset_return_mean_ratings(train_set, itm[0])

        if norm >= global_mean_user:
            list_temp_binario = [int(itm[0]), int(itm[1]), 1]  # ENTRA COMO 1 NA MATRIZ BINÁRIA (DADOS IMPLÍCITOS)
        else:
            list_temp_binario = [int(itm[0]), int(itm[1]), 0]  # ENTRA COMO 0 NA MATRIZ BINÁRIA (DADOS IMPLÍCITOS)
        listinha_do_binario.append(list_temp_binario)

        if cycle % 100 == 0:
            print("Cycle: {}/{}".format(cycle, size))

        cycle += 1

    df_to_concat = pd.DataFrame(listinha)  # matrix with the mean ratings
    df_binario = pd.DataFrame(listinha_do_binario)  # matrix with the binary criteria

    return df_to_concat, df_binario


# Binarization - test_set
def binarization_test_set(test_file, threshold, sep)::

    test_set_in = pd.read_csv(test_file, sep=sep, header=None)
    test_file_dict = ReadFile(test_file, sep=sep).read()

    listinha_threshold = []
    listinha_mean = []

    for row, itm in test_set_in.iterrows():

        # Caso a nota seja maior ou igual ao threshold, recebe 1 (notas implícitas). Senão, 0
        if itm[2] >= threshold:
            implicit_rating_threshold = 1
        else:
            implicit_rating_threshold = 0
        listinha_temp_threshold = [int(itm[0]), int(itm[1]), int(implicit_rating_threshold)]

        if itm[2] >= get_user_and_trainset_return_mean_ratings(test_file_dict, itm[0]):
            implicit_rating_mean = 1
        else:
            implicit_rating_mean = 0
        listinha_temp_mean = [int(itm[0]), int(itm[1]), int(implicit_rating_mean)]

        listinha_threshold.append(listinha_temp_threshold)
        listinha_mean.append(listinha_temp_mean)

    df_binario_threshold = pd.DataFrame(listinha_threshold)  # matrix with the threshold criteria
    df_binario_media = pd.DataFrame(listinha_mean)  # matrix with the mean criteria

    return df_binario_threshold, df_binario_media


# Determina a quantidade de ZEROS e UNS, calcula a proporção entre eles
def calc_zeros_and_ones(matrix):
    print("************************")
    size_all = len(matrix[2])
    print("LEN matrix[2]:", len(matrix[2]))
    number_zeros_in_ratings = (matrix[2] == 0).astype(int).sum(axis=0)
    number_ones_in_ratings = (matrix[2] == 1).astype(int).sum(axis=0)

    print("% ZEROS IN BINARY MATRIX:", (number_zeros_in_ratings/size_all)*100)
    print("% ONES IN BINARY MATRIX:", (number_ones_in_ratings/size_all)*100)

    print("************************")
