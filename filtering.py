import pandas as pd

"""
Recebe duas matrizes de dados:
- uma matriz binária (que passou pelo framework e foram unidos (Lt1, Lt2)): main2_df_binario_threshold.dat e main2_df_binario_media.dat
- a train.dat, que passou pelo LTR, fez a recomendação e mandou um output (ranking_file.dat)

Faz a filtragem deste ranking_file.dat, verificando se está 1 ou 0 na main2_df_binario_threshold/media.dat
"""


def filter_(binary_matrix_file, ranking_file, sep='\t', delete=0):

    """
    Recebe dois arquivos, da matriz binarizada e um de ranking. Faz a leitura de cada item da matriz binária,
    dado o critério em delete, do item que será apagado da ranking_file.
    Se apagar os ZEROS, manterão os UNS, ou seja, os mais confiantes e passará direto pela avaliação.
    Se apagar os UNS, mantendo os ZEROS, passarão depois pelo algoritmo de recomendação

    :param binary_matrix_file: arquivo com valores que foram binarizados da matriz
    :param ranking_file: arquivo de saída do ItemKNN, rankings dos itens para cada usuário
    :param sep: Separador padrão tab
    :param delete: exclui ZERO ou UM
    :return: index dos itens a serm apagados da ranking_file
    """

    binary_matrix = pd.read_csv(binary_matrix_file, sep=sep, header=None, encoding="utf-8-sig")
    ranking = pd.read_csv(ranking_file, sep=sep, header=None, encoding="utf-8-sig")

    if delete == 0:
        binary_matrix = binary_matrix.loc[binary_matrix[2] == 0]  # Matriz com as linhas que contém ZERO na terceira coluna, que serão excluídas da matriz ranking
    elif delete == 1:
        binary_matrix = binary_matrix.loc[binary_matrix[2] == 1]  # Matriz com as linhas que contém UM na terceira coluna, que serão excluídas da matriz ranking
    else:
        print("ERROR! 'delete' parameter should be 1 or 0!")

    cycle = 1
    size = len(binary_matrix)

    list_to_drop = list()

    for row, itm in binary_matrix.iterrows():

        if index_list_to_drop:  # check if the list is not empty
            list_to_drop.append(index_list_to_drop[0])

        print("Cycle: {}/{}. Len list_to_drop: {}".format(cycle, size, len(list_to_drop)))

        cycle += 1

    return ranking
