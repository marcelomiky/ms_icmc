from caserec.utils.split_database import SplitDatabase
from caserec.recommenders.rating_prediction.corec import ECoRec
from caserec.recommenders.item_recommendation.userknn import UserKNN
from caserec.recommenders.item_recommendation.itemknn import ItemKNN
from caserec.recommenders.item_recommendation.bprmf import BprMF
from caserec.recommenders.item_recommendation.most_popular import MostPopular
from caserec.recommenders.item_recommendation.paco_recommender import PaCoRecommender
from caserec.generate_dataset_labeled_etc import gen_confident_and_enriched_file



def run_corec_10_fold(param, n_sample, train_file_in, test_file_in, ensemble_in, rec1, rec2):

    print("\nRunning Corec! confidence_measure = {} - number_sample = {} - ensemble_method = {} ***".format(param, n_sample, ensemble_in))
    ecorec = ECoRec(train_file=train_file_in, test_file=test_file_in,
                    recommenders=(rec1, rec2),
                    confidence_measure=param,
                    number_sample=n_sample,
                    ensemble_method=ensemble_in)
    ecorec.compute()
    print("\n")
    print("\nFIM!")



def run_corec(param, n_sample, train_file_in, test_file_in, ensemble_in, rec1, rec2):

    print("\nRunning Corec! confidence_measure = {} - number_sample = {} - ensemble_method = {} ***".format(param, n_sample, ensemble_in))
    ecorec = ECoRec(train_file=train_file_in, test_file=test_file_in,
                    recommenders=(rec1, rec2),
                    confidence_measure=param,
                    number_sample=n_sample,
                    ensemble_method=ensemble_in)
    ecorec.compute()
    print("\n")
    print("\nFIM!")


def user_knn_here(as_binary_in, rank_length_in, train_file_baseline, test_file,
                  train_file_labeled_set_A, train_file_labeled_set_B, train_file_ensemble_set):

    print("\n*** BASELINE! Algorithm: UserKNN ***\n")
    UserKNN(train_file=train_file_baseline, test_file=test_file, as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** labeled_set_A! Algorithm: UserKNN ***\n")
    UserKNN(train_file=train_file_labeled_set_A, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** labeled_set_B! Algorithm: UserKNN ***\n")
    UserKNN(train_file=train_file_labeled_set_B, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** ensemble_set! Algorithm: UserKNN ***\n")
    UserKNN(train_file=train_file_ensemble_set, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)


def item_knn_here(as_binary_in, rank_length_in, train_file_baseline, test_file,
                  train_file_labeled_set_A, train_file_labeled_set_B, train_file_ensemble_set):

    print("\n*** BASELINE! Algorithm: ItemKNN ***\n")
    ItemKNN(train_file=train_file_baseline, test_file=test_file, as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** labeled_set_A! Algorithm: ItemKNN ***\n")
    ItemKNN(train_file=train_file_labeled_set_A, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** labeled_set_B! Algorithm: ItemKNN ***\n")
    ItemKNN(train_file=train_file_labeled_set_B, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)

    print("\n*** ensemble_set! Algorithm: ItemKNN ***\n")
    ItemKNN(train_file=train_file_ensemble_set, test_file=test_file,
            as_binary=as_binary_in,
            rank_length=rank_length_in).compute(as_table=True)



def bprmf_here(train_file_baseline, test_file, output_file_baseline, output_file_ensemble_list,
               train_file_ensemble_set_list, number_sample_list, lr, epoch, rl):

    print("\n*** BASELINE! Algorithm: BPRMF ***\n")
    BprMF(train_file=train_file_baseline, test_file=test_file, output_file=output_file_baseline,
          learn_rate=lr, epochs=epoch, rank_length=rl).compute(metrics=['MAP', 'NDCG'], as_table=True)

    for i in range(len(train_file_ensemble_set_list)):
        print("\n*** ENSEMBLE_SET! Algorithm: BPRMF - "
              "_n_sample = {} - i = {}/{} ***\n".format(number_sample_list[i], i+1, len(train_file_ensemble_set_list)))
        BprMF(train_file=train_file_ensemble_set_list[i],
              test_file=test_file,
              output_file=output_file_ensemble_list[i],
              learn_rate=lr, epochs=epoch, rank_length=rl).compute(metrics=['MAP', 'NDCG'], as_table=True)


def bprmf_here_new(train_file_baseline, test_file, output_file_baseline, train_file_ensemble_set, output_file_ensemble,
                   lr, epoch, rl):

    print("\n*** BASELINE! Algorithm: BPRMF ***\n")
    BprMF(train_file=train_file_baseline, test_file=test_file, output_file=output_file_baseline,
          learn_rate=lr,
          epochs=epoch,
          rank_length=rl).compute(metrics=['MAP', 'NDCG'], as_table=True)

    print("\n*** ensemble_set! Algorithm: BPRMF ***\n")
    BprMF(train_file=train_file_ensemble_set, test_file=test_file, output_file=output_file_ensemble,
          learn_rate=lr,
          epochs=epoch,
          rank_length=rl).compute(metrics=['MAP', 'NDCG'], as_table=True)


def most_popular_here(train_file_baseline, test_file, train_file_labeled_set_A, train_file_labeled_set_B,
                      train_file_ensemble_set):

    print("\n*** BASELINE! Algorithm: Most Popular ***\n")
    MostPopular(train_file=train_file_baseline, test_file=test_file).compute(as_table=True)

    print("\n*** labeled_set_A! Algorithm: Most Popular ***\n")
    MostPopular(train_file=train_file_labeled_set_A, test_file=test_file).compute(as_table=True)

    print("\n*** labeled_set_B! Algorithm: Most Popular ***\n")
    MostPopular(train_file=train_file_labeled_set_B, test_file=test_file).compute(as_table=True)

    print("\n*** ensemble_set! Algorithm: Most Popular ***\n")
    MostPopular(train_file=train_file_ensemble_set, test_file=test_file).compute(as_table=True)


def paco_here(train_file_baseline, test_file, train_file_labeled_set_A, train_file_labeled_set_B,
              train_file_ensemble_set):

    print("\n*** BASELINE! Algorithm: PaCo ***\n")
    PaCoRecommender(train_file=train_file_baseline, test_file=test_file).compute(as_table=True)

    print("\n*** labeled_set_A! Algorithm: PaCo ***\n")
    PaCoRecommender(train_file=train_file_labeled_set_A, test_file=test_file).compute(as_table=True)

    print("\n*** labeled_set_B! Algorithm: PaCo ***\n")
    PaCoRecommender(train_file=train_file_labeled_set_B, test_file=test_file).compute(as_table=True)

    print("\n*** ensemble_set! Algorithm: PaCo ***\n")
    PaCoRecommender(train_file=train_file_ensemble_set, test_file=test_file).compute(as_table=True)

def paco_here_new(train_file_ensemble_set, test_file, min_density, density_low):

    print("\n*** ensemble_set! Algorithm: PaCo ***\n")
    PaCoRecommender(train_file=train_file_ensemble_set, test_file=test_file,
                    min_density=min_density,
                    density_low=density_low).compute(metrics=['MAP', 'NDCG'], as_table=True)


### FILMTRUST
# Baseline Dataset
# train_file_baseline = 'datasets/filmtrust/0 folds/train.dat'
# test_file = 'datasets/filmtrust/0 folds/test.dat'
# ranking_file_userknn = 'datasets/filmtrust/0 folds/ranking_file_userknn_v0.dat'
# ranking_file_itemknn = 'datasets/filmtrust/0 folds/ranking_file_itemknn_v0.dat'
#
# # Labeled Set 1 Dataset
# train_file_labeled_set_1 = 'datasets/filmtrust/0 folds/labeled_set_1.dat'
# ranking_file_userknn_labeled_set_1 = 'datasets/filmtrust/0 folds/ranking_file_userknn_v1_labeled_set1.dat'
# ranking_file_itemknn_labeled_set_1 = 'datasets/filmtrust/0 folds/ranking_file_itemknn_v1_labeled_set1.dat'
#
# # Labeled Set 2 Dataset
# train_file_labeled_set_2 = 'datasets/filmtrust/0 folds/labeled_set_2.dat'
# ranking_file_userknn_labeled_set_2 = 'datasets/filmtrust/0 folds/ranking_file_userknn_v1_labeled_set2.dat'
# ranking_file_itemknn_labeled_set_2 = 'datasets/filmtrust/0 folds/ranking_file_itemknn_v1_labeled_set2.dat'
#
# # Propose Dataset
# train_file_ensemble_set = 'datasets/filmtrust/0 folds/ensemble_set.dat'
# ranking_file_userknn_ensemble_set = 'datasets/filmtrust/0 folds/ranking_file_userknn_v2_ensemble_set.dat'
# ranking_file_itemknn_ensemble_set = 'datasets/filmtrust/0 folds/ranking_file_itemknn_v2_ensemble_set.dat'

### Bookcrossing
# Baseline Dataset
# BookCrossing Dataset
# train_file_baseline = 'datasets/BookCrossing/folds/0/train.dat'
# test_file = 'datasets/BookCrossing/folds/0/test.dat'
# ranking_file_userknn = 'datasets/BookCrossing/folds/0/ranking_file_userknn_v0.dat'
# ranking_file_itemknn = 'datasets/BookCrossing/folds/0/ranking_file_itemknn_v0.dat'
#
# # Labeled Set 1 Dataset
# train_file_labeled_set_1 = 'datasets/BookCrossing/folds/0/labeled_set_1.dat'
# ranking_file_userknn_labeled_set_1 = 'datasets/BookCrossing/folds/0/ranking_file_userknn_v1_labeled_set1.dat'
# ranking_file_itemknn_labeled_set_1 = 'datasets/BookCrossing/folds/0/ranking_file_itemknn_v1_labeled_set1.dat'
#
# # Labeled Set 2 Dataset
# train_file_labeled_set_2 = 'datasets/BookCrossing/folds/0/labeled_set_2.dat'
# ranking_file_userknn_labeled_set_2 = 'datasets/BookCrossing/folds/0/ranking_file_userknn_v1_labeled_set2.dat'
# ranking_file_itemknn_labeled_set_2 = 'datasets/BookCrossing/folds/0/ranking_file_itemknn_v1_labeled_set2.dat'
#
# # Propose Dataset
# train_file_ensemble_set = 'datasets/BookCrossing/folds/0/ensemble_set.dat'
# ranking_file_userknn_ensemble_set = 'datasets/BookCrossing/folds/0/ranking_file_userknn_v2_ensemble_set.dat'
# ranking_file_itemknn_ensemble_set = 'datasets/BookCrossing/folds/0/ranking_file_itemknn_v2_ensemble_set.dat'


### Movielens-2k
# Baseline Dataset
# train_file_baseline = 'datasets/movielens-2k/folds/0/train.dat'
# test_file = 'datasets/movielens-2k/folds/0/test.dat'
# ranking_file_userknn = 'datasets/movielens-2k/folds/0/ranking_file_userknn_v0.dat'
# ranking_file_itemknn = 'datasets/movielens-2k/folds/0/ranking_file_itemknn_v0.dat'
#
# # Labeled Set 1 Dataset
# train_file_labeled_set_1 = 'datasets/movielens-2k/folds/0/labeled_set_1.dat'
# ranking_file_userknn_labeled_set_1 = 'datasets/movielens-2k/folds/0/ranking_file_userknn_v1_labeled_set1.dat'
# ranking_file_itemknn_labeled_set_1 = 'datasets/movielens-2k/folds/0/ranking_file_itemknn_v1_labeled_set1.dat'
#
# # Labeled Set 2 Dataset
# train_file_labeled_set_2 = 'datasets/movielens-2k/folds/0/labeled_set_2.dat'
# ranking_file_userknn_labeled_set_2 = 'datasets/movielens-2k/folds/0/ranking_file_userknn_v1_labeled_set2.dat'
# ranking_file_itemknn_labeled_set_2 = 'datasets/movielens-2k/folds/0/ranking_file_itemknn_v1_labeled_set2.dat'
#
# # Propose Dataset
# train_file_ensemble_set = 'datasets/movielens-2k/folds/0/ensemble_set.dat'
# ranking_file_userknn_ensemble_set = 'datasets/movielens-2k/folds/0/ranking_file_userknn_v2_ensemble_set.dat'
# ranking_file_itemknn_ensemble_set = 'datasets/movielens-2k/folds/0/ranking_file_itemknn_v2_ensemble_set.dat'



def run_corec_10_fold(param, n_sample, ensemble_in):
    train_file0 = 'datasets/filmtrust/folds/0/train.dat'
    train_file1 = 'datasets/filmtrust/folds/1/train.dat'
    train_file2 = 'datasets/filmtrust/folds/2/train.dat'
    train_file3 = 'datasets/filmtrust/folds/3/train.dat'
    train_file4 = 'datasets/filmtrust/folds/4/train.dat'
    train_file5 = 'datasets/filmtrust/folds/5/train.dat'
    train_file6 = 'datasets/filmtrust/folds/6/train.dat'
    train_file7 = 'datasets/filmtrust/folds/7/train.dat'
    train_file8 = 'datasets/filmtrust/folds/8/train.dat'
    train_file9 = 'datasets/filmtrust/folds/9/train.dat'

    list_train_file = [train_file0, train_file1, train_file2, train_file3, train_file4, train_file5, train_file6,
                       train_file7, train_file8, train_file9]

    test_file0 = 'datasets/filmtrust/folds/0/test.dat'
    test_file1 = 'datasets/filmtrust/folds/1/test.dat'
    test_file2 = 'datasets/filmtrust/folds/2/test.dat'
    test_file3 = 'datasets/filmtrust/folds/3/test.dat'
    test_file4 = 'datasets/filmtrust/folds/4/test.dat'
    test_file5 = 'datasets/filmtrust/folds/5/test.dat'
    test_file6 = 'datasets/filmtrust/folds/6/test.dat'
    test_file7 = 'datasets/filmtrust/folds/7/test.dat'
    test_file8 = 'datasets/filmtrust/folds/8/test.dat'
    test_file9 = 'datasets/filmtrust/folds/9/test.dat'

    list_test_file = [test_file0, test_file1, test_file2, test_file3, test_file4, test_file5, test_file6, test_file7,
                      test_file8, test_file9]

    for i in range(10):
        print("\nRunning Corec! confidence_measure = {} - "
              "number_sample = {} - ensemble_method = {}"
              "*** i:{}/9 (0 ~ 9) ***".format(param, n_sample, ensemble_in, i))

        ecorec = ECoRec(train_file=list_train_file[i], test_file=list_test_file[i],
                        recommenders=(1, 2),
                        confidence_measure=param,
                        number_sample=n_sample,
                        ensemble_method=ensemble_in)
        ecorec.compute()
        print("\n")

    print("\nFIM!")


################### Binarização para o UserKNN ####################

def baseline(as_binary_in):
    train_file0 = 'datasets/filmtrust/folds/0/train.dat'
    train_file1 = 'datasets/filmtrust/folds/1/train.dat'
    train_file2 = 'datasets/filmtrust/folds/2/train.dat'
    train_file3 = 'datasets/filmtrust/folds/3/train.dat'
    train_file4 = 'datasets/filmtrust/folds/4/train.dat'
    train_file5 = 'datasets/filmtrust/folds/5/train.dat'
    train_file6 = 'datasets/filmtrust/folds/6/train.dat'
    train_file7 = 'datasets/filmtrust/folds/7/train.dat'
    train_file8 = 'datasets/filmtrust/folds/8/train.dat'
    train_file9 = 'datasets/filmtrust/folds/9/train.dat'

    list_train_file = [train_file0, train_file1, train_file2, train_file3, train_file4,
                       train_file5, train_file6, train_file7, train_file8, train_file9]

    test_file0 = 'datasets/filmtrust/folds/0/test.dat'
    test_file1 = 'datasets/filmtrust/folds/1/test.dat'
    test_file2 = 'datasets/filmtrust/folds/2/test.dat'
    test_file3 = 'datasets/filmtrust/folds/3/test.dat'
    test_file4 = 'datasets/filmtrust/folds/4/test.dat'
    test_file5 = 'datasets/filmtrust/folds/5/test.dat'
    test_file6 = 'datasets/filmtrust/folds/6/test.dat'
    test_file7 = 'datasets/filmtrust/folds/7/test.dat'
    test_file8 = 'datasets/filmtrust/folds/8/test.dat'
    test_file9 = 'datasets/filmtrust/folds/9/test.dat'

    list_test_file = [test_file0, test_file1, test_file2, test_file3, test_file4,
                      test_file5, test_file6, test_file7, test_file8, test_file9]

    ranking_file_userknn0 = 'datasets/filmtrust/folds/0/ranking_file_userknn_v0.dat'
    ranking_file_userknn1 = 'datasets/filmtrust/folds/1/ranking_file_userknn_v0.dat'
    ranking_file_userknn2 = 'datasets/filmtrust/folds/2/ranking_file_userknn_v0.dat'
    ranking_file_userknn3 = 'datasets/filmtrust/folds/3/ranking_file_userknn_v0.dat'
    ranking_file_userknn4 = 'datasets/filmtrust/folds/4/ranking_file_userknn_v0.dat'
    ranking_file_userknn5 = 'datasets/filmtrust/folds/5/ranking_file_userknn_v0.dat'
    ranking_file_userknn6 = 'datasets/filmtrust/folds/6/ranking_file_userknn_v0.dat'
    ranking_file_userknn7 = 'datasets/filmtrust/folds/7/ranking_file_userknn_v0.dat'
    ranking_file_userknn8 = 'datasets/filmtrust/folds/8/ranking_file_userknn_v0.dat'
    ranking_file_userknn9 = 'datasets/filmtrust/folds/9/ranking_file_userknn_v0.dat'

    list_ranking_file = [ranking_file_userknn0, ranking_file_userknn1, ranking_file_userknn2, ranking_file_userknn3,
                         ranking_file_userknn4, ranking_file_userknn5, ranking_file_userknn6, ranking_file_userknn7,
                         ranking_file_userknn8, ranking_file_userknn9]

    for i in range(10):
        print("\n*** BASELINE! i:{}/9 (0 ~ 9) - as_binary = {} ***".format(i, as_binary_in))
        UserKNN(train_file=list_train_file[i],
                test_file=list_test_file[i],
                as_binary=as_binary_in,
                output_file=list_ranking_file[i],
                rank_length=100).compute(as_table=True)
        print("\n")



def user_knn_labeled_set_1(as_binary_in):

    labeled_set_1_0 = 'datasets/filmtrust/folds/0/labeled_set_1.dat'
    labeled_set_1_1 = 'datasets/filmtrust/folds/1/labeled_set_1.dat'
    labeled_set_1_2 = 'datasets/filmtrust/folds/2/labeled_set_1.dat'
    labeled_set_1_3 = 'datasets/filmtrust/folds/3/labeled_set_1.dat'
    labeled_set_1_4 = 'datasets/filmtrust/folds/4/labeled_set_1.dat'
    labeled_set_1_5 = 'datasets/filmtrust/folds/5/labeled_set_1.dat'
    labeled_set_1_6 = 'datasets/filmtrust/folds/6/labeled_set_1.dat'
    labeled_set_1_7 = 'datasets/filmtrust/folds/7/labeled_set_1.dat'
    labeled_set_1_8 = 'datasets/filmtrust/folds/8/labeled_set_1.dat'
    labeled_set_1_9 = 'datasets/filmtrust/folds/9/labeled_set_1.dat'

    list_train_file = [labeled_set_1_0, labeled_set_1_1, labeled_set_1_2, labeled_set_1_3, labeled_set_1_4,
                       labeled_set_1_5, labeled_set_1_6, labeled_set_1_7, labeled_set_1_8, labeled_set_1_9]

    test_file_0 = 'datasets/filmtrust/folds/0/test.dat'
    test_file_1 = 'datasets/filmtrust/folds/1/test.dat'
    test_file_2 = 'datasets/filmtrust/folds/2/test.dat'
    test_file_3 = 'datasets/filmtrust/folds/3/test.dat'
    test_file_4 = 'datasets/filmtrust/folds/4/test.dat'
    test_file_5 = 'datasets/filmtrust/folds/5/test.dat'
    test_file_6 = 'datasets/filmtrust/folds/6/test.dat'
    test_file_7 = 'datasets/filmtrust/folds/7/test.dat'
    test_file_8 = 'datasets/filmtrust/folds/8/test.dat'
    test_file_9 = 'datasets/filmtrust/folds/9/test.dat'

    list_test_file = [test_file_0, test_file_1, test_file_2, test_file_3, test_file_4,
                      test_file_5, test_file_6, test_file_7, test_file_8, test_file_9]

    ranking_file_userknn0 = 'datasets/filmtrust/folds/0/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn1 = 'datasets/filmtrust/folds/1/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn2 = 'datasets/filmtrust/folds/2/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn3 = 'datasets/filmtrust/folds/3/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn4 = 'datasets/filmtrust/folds/4/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn5 = 'datasets/filmtrust/folds/5/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn6 = 'datasets/filmtrust/folds/6/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn7 = 'datasets/filmtrust/folds/7/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn8 = 'datasets/filmtrust/folds/8/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn9 = 'datasets/filmtrust/folds/9/ranking_file_userknn_v1_set2.dat'

    list_ranking_file = [ranking_file_userknn0, ranking_file_userknn1, ranking_file_userknn2, ranking_file_userknn3,
                         ranking_file_userknn4, ranking_file_userknn5, ranking_file_userknn6, ranking_file_userknn7,
                         ranking_file_userknn8, ranking_file_userknn9]

    for i in range(10):
        print("\n***labeled_set_1 *** i:{}/9 (0 ~ 9) - as_binary = {} ***".format(i, as_binary_in))
        UserKNN(train_file=list_train_file[i],
                test_file=list_test_file[i],
                as_binary=as_binary_in,
                output_file=list_ranking_file[i],
                rank_length=100).compute(as_table=True)
        print("\n")

def user_knn_labeled_set_2(as_binary_in):

    labeled_set_2_0 = 'datasets/filmtrust/folds/0/labeled_set_2.dat'
    labeled_set_2_1 = 'datasets/filmtrust/folds/1/labeled_set_2.dat'
    labeled_set_2_2 = 'datasets/filmtrust/folds/2/labeled_set_2.dat'
    labeled_set_2_3 = 'datasets/filmtrust/folds/3/labeled_set_2.dat'
    labeled_set_2_4 = 'datasets/filmtrust/folds/4/labeled_set_2.dat'
    labeled_set_2_5 = 'datasets/filmtrust/folds/5/labeled_set_2.dat'
    labeled_set_2_6 = 'datasets/filmtrust/folds/6/labeled_set_2.dat'
    labeled_set_2_7 = 'datasets/filmtrust/folds/7/labeled_set_2.dat'
    labeled_set_2_8 = 'datasets/filmtrust/folds/8/labeled_set_2.dat'
    labeled_set_2_9 = 'datasets/filmtrust/folds/9/labeled_set_2.dat'

    list_train_file = [labeled_set_2_0, labeled_set_2_1, labeled_set_2_2, labeled_set_2_3, labeled_set_2_4,
                       labeled_set_2_5, labeled_set_2_6, labeled_set_2_7, labeled_set_2_8, labeled_set_2_9]

    test_file_0 = 'datasets/filmtrust/folds/0/test.dat'
    test_file_1 = 'datasets/filmtrust/folds/1/test.dat'
    test_file_2 = 'datasets/filmtrust/folds/2/test.dat'
    test_file_3 = 'datasets/filmtrust/folds/3/test.dat'
    test_file_4 = 'datasets/filmtrust/folds/4/test.dat'
    test_file_5 = 'datasets/filmtrust/folds/5/test.dat'
    test_file_6 = 'datasets/filmtrust/folds/6/test.dat'
    test_file_7 = 'datasets/filmtrust/folds/7/test.dat'
    test_file_8 = 'datasets/filmtrust/folds/8/test.dat'
    test_file_9 = 'datasets/filmtrust/folds/9/test.dat'

    list_test_file = [test_file_0, test_file_1, test_file_2, test_file_3, test_file_4, test_file_5,
                      test_file_6, test_file_7, test_file_8, test_file_9]

    ranking_file_userknn0 = 'datasets/filmtrust/folds/0/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn1 = 'datasets/filmtrust/folds/1/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn2 = 'datasets/filmtrust/folds/2/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn3 = 'datasets/filmtrust/folds/3/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn4 = 'datasets/filmtrust/folds/4/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn5 = 'datasets/filmtrust/folds/5/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn6 = 'datasets/filmtrust/folds/6/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn7 = 'datasets/filmtrust/folds/7/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn8 = 'datasets/filmtrust/folds/8/ranking_file_userknn_v1_set2.dat'
    ranking_file_userknn9 = 'datasets/filmtrust/folds/9/ranking_file_userknn_v1_set2.dat'

    list_ranking_file = [ranking_file_userknn0, ranking_file_userknn1, ranking_file_userknn2, ranking_file_userknn3,
                         ranking_file_userknn4, ranking_file_userknn5, ranking_file_userknn6, ranking_file_userknn7,
                         ranking_file_userknn8, ranking_file_userknn9]

    for i in range(10):
        print("\n***labeled_set_2 *** i:{}/9 (0 ~ 9) - as_binary = {} ***".format(i, as_binary_in))
        UserKNN(train_file=list_train_file[i],
                test_file=list_test_file[i],
                as_binary=as_binary_in,
                output_file=list_ranking_file[i],
                rank_length=100).compute(as_table=True)
        print("\n")



# ############### FUNÇÃO PARA RODAR O MODELO COM OS DADOS BINARIZADOS ###########################
def run_model(as_binary_in):

    train_file_0 = 'datasets/filmtrust/folds/0/ensemble_set.dat'
    train_file_1 = 'datasets/filmtrust/folds/1/ensemble_set.dat'
    train_file_2 = 'datasets/filmtrust/folds/2/ensemble_set.dat'
    train_file_3 = 'datasets/filmtrust/folds/3/ensemble_set.dat'
    train_file_4 = 'datasets/filmtrust/folds/4/ensemble_set.dat'
    train_file_5 = 'datasets/filmtrust/folds/5/ensemble_set.dat'
    train_file_6 = 'datasets/filmtrust/folds/6/ensemble_set.dat'
    train_file_7 = 'datasets/filmtrust/folds/7/ensemble_set.dat'
    train_file_8 = 'datasets/filmtrust/folds/8/ensemble_set.dat'
    train_file_9 = 'datasets/filmtrust/folds/9/ensemble_set.dat'

    list_train_file = [train_file_0, train_file_1, train_file_2, train_file_3, train_file_4, train_file_5,
                       train_file_6, train_file_7, train_file_8, train_file_9]

    test_file_0 = 'datasets/filmtrust/folds/0/test.dat'
    test_file_1 = 'datasets/filmtrust/folds/1/test.dat'
    test_file_2 = 'datasets/filmtrust/folds/2/test.dat'
    test_file_3 = 'datasets/filmtrust/folds/3/test.dat'
    test_file_4 = 'datasets/filmtrust/folds/4/test.dat'
    test_file_5 = 'datasets/filmtrust/folds/5/test.dat'
    test_file_6 = 'datasets/filmtrust/folds/6/test.dat'
    test_file_7 = 'datasets/filmtrust/folds/7/test.dat'
    test_file_8 = 'datasets/filmtrust/folds/8/test.dat'
    test_file_9 = 'datasets/filmtrust/folds/9/test.dat'

    list_test_file = [test_file_0, test_file_1, test_file_2, test_file_3, test_file_4, test_file_5,
                      test_file_6, test_file_7, test_file_8, test_file_9]

    ranking_file_userknn0 = 'datasets/filmtrust/folds/0/ranking_file_userknn_v2.dat'
    ranking_file_userknn1 = 'datasets/filmtrust/folds/1/ranking_file_userknn_v2.dat'
    ranking_file_userknn2 = 'datasets/filmtrust/folds/2/ranking_file_userknn_v2.dat'
    ranking_file_userknn3 = 'datasets/filmtrust/folds/3/ranking_file_userknn_v2.dat'
    ranking_file_userknn4 = 'datasets/filmtrust/folds/4/ranking_file_userknn_v2.dat'
    ranking_file_userknn5 = 'datasets/filmtrust/folds/5/ranking_file_userknn_v2.dat'
    ranking_file_userknn6 = 'datasets/filmtrust/folds/6/ranking_file_userknn_v2.dat'
    ranking_file_userknn7 = 'datasets/filmtrust/folds/7/ranking_file_userknn_v2.dat'
    ranking_file_userknn8 = 'datasets/filmtrust/folds/8/ranking_file_userknn_v2.dat'
    ranking_file_userknn9 = 'datasets/filmtrust/folds/9/ranking_file_userknn_v2.dat'

    list_ranking_file = [ranking_file_userknn0, ranking_file_userknn1, ranking_file_userknn2, ranking_file_userknn3,
                         ranking_file_userknn4, ranking_file_userknn5, ranking_file_userknn6, ranking_file_userknn7,
                         ranking_file_userknn8, ranking_file_userknn9]

    for i in range(10):
        print("\n*** Running Model i:{}/9 (0 ~ 9)***".format(i))
        UserKNN(train_file=list_train_file[i],
                test_file=list_test_file[i],
                as_binary=as_binary_in,
                output_file=list_ranking_file[i],
                rank_length=100).compute(as_table=True)
        print("\n")




def baseline():

    train_file0 = 'datasets/filmtrust/train.dat'
    test_file0 = 'datasets/filmtrust/test.dat'
    ranking_file_itemknn0 = 'datasets/filmtrust/ranking_file_itemknn_v0.dat'

    print("\n*** BASELINE!")
    ItemKNN(train_file=train_file0,
            test_file=test_file0,
            as_binary=False,
            output_file=ranking_file_itemknn0,
            rank_length=100).compute(as_table=True)
    print("\n")

def item_knn_labeled_set_1():

    train_file0 = 'datasets/filmtrust/labeled_set_1.dat'
    test_file0 = 'datasets/filmtrust/test.dat'
    ranking_file_itemknn0 = 'datasets/filmtrust/ranking_file_itemknn_v1_set1_inverted_confidence_metric.dat'

    print("\nlabeled_set_1: ***")
    ItemKNN(train_file=train_file0, test_file=test_file0, as_binary=False,
            output_file=ranking_file_itemknn0, rank_length=100).compute(as_table=True)
    print("\n")

def item_knn_labeled_set_2():

    train_file0 = 'datasets/filmtrust/folds/0/labeled_set_2.dat'
    test_file0 = 'datasets/filmtrust/0 folds/test.dat'
    ranking_file_itemknn0 = 'datasets/filmtrust/0 folds/ranking_file_itemknn_v1_set2_inverted_confidence_metric.dat'

    print("\nlabeled_set_2: ***")

    ItemKNN(train_file=train_file0, test_file=test_file0, as_binary=False,
            output_file=ranking_file_itemknn0, rank_length=100).compute(as_table=True)
    print("\n")


def run_model():

    # train_file0 = '../../datasets/filmtrust/folds/0/train.dat'
    # train_file1 = '../../datasets/filmtrust/folds/1/train.dat'
    # train_file2 = '../../datasets/filmtrust/folds/2/train.dat'
    # train_file3 = '../../datasets/filmtrust/folds/3/train.dat'
    # train_file4 = '../../datasets/filmtrust/folds/4/train.dat'
    # train_file5 = '../../datasets/filmtrust/folds/5/train.dat'
    # train_file6 = '../../datasets/filmtrust/folds/6/train.dat'
    # train_file7 = '../../datasets/filmtrust/folds/7/train.dat'
    # train_file8 = '../../datasets/filmtrust/folds/8/train.dat'
    # train_file9 = '../../datasets/filmtrust/folds/9/train.dat'
    #
    # list_train_file = [train_file0, train_file1, train_file2, train_file3, train_file4, train_file5, train_file6,
    #                    train_file7, train_file8, train_file9]
    #
    # test_file0 = '../../datasets/filmtrust/folds/0/test.dat'
    # test_file1 = '../../datasets/filmtrust/folds/1/test.dat'
    # test_file2 = '../../datasets/filmtrust/folds/2/test.dat'
    # test_file3 = '../../datasets/filmtrust/folds/3/test.dat'
    # test_file4 = '../../datasets/filmtrust/folds/4/test.dat'
    # test_file5 = '../../datasets/filmtrust/folds/5/test.dat'
    # test_file6 = '../../datasets/filmtrust/folds/6/test.dat'
    # test_file7 = '../../datasets/filmtrust/folds/7/test.dat'
    # test_file8 = '../../datasets/filmtrust/folds/8/test.dat'
    # test_file9 = '../../datasets/filmtrust/folds/9/test.dat'
    #
    # list_test_file = [test_file0, test_file1, test_file2, test_file3, test_file4, test_file5, test_file6, test_file7,
    #                   test_file8, test_file9]
    #
    # train_plus_enriched0 = '../../datasets/filmtrust/folds/0/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched1 = '../../datasets/filmtrust/folds/1/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched2 = '../../datasets/filmtrust/folds/2/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched3 = '../../datasets/filmtrust/folds/3/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched4 = '../../datasets/filmtrust/folds/4/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched5 = '../../datasets/filmtrust/folds/5/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched6 = '../../datasets/filmtrust/folds/6/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched7 = '../../datasets/filmtrust/folds/7/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched8 = '../../datasets/filmtrust/folds/8/train_plus_enriched_only_when_has_intersection.dat'
    # train_plus_enriched9 = '../../datasets/filmtrust/folds/9/train_plus_enriched_only_when_has_intersection.dat'
    #
    # list_train_plus_enriched = [train_plus_enriched0, train_plus_enriched1, train_plus_enriched2, train_plus_enriched3,
    #                             train_plus_enriched4, train_plus_enriched5, train_plus_enriched6, train_plus_enriched7,
    #                             train_plus_enriched8, train_plus_enriched9]
    #
    # ranking_file_itemknn0 = '../../datasets/filmtrust/folds/0/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn1 = '../../datasets/filmtrust/folds/1/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_ItemKNN = '../../datasets/filmtrust/folds/2/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn3 = '../../datasets/filmtrust/folds/3/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn4 = '../../datasets/filmtrust/folds/4/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn5 = '../../datasets/filmtrust/folds/5/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn6 = '../../datasets/filmtrust/folds/6/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn7 = '../../datasets/filmtrust/folds/7/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn8 = '../../datasets/filmtrust/folds/8/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    # ranking_file_itemknn9 = '../../datasets/filmtrust/folds/9/ranking_file_itemknn_v3_inverted_confidence_metric.dat'
    #
    # list_ranking_file = [ranking_file_itemknn0, ranking_file_itemknn1, ranking_file_ItemKNN, ranking_file_itemknn3,
    #                      ranking_file_itemknn4, ranking_file_itemknn5, ranking_file_itemknn6, ranking_file_itemknn7,
    #                      ranking_file_itemknn8, ranking_file_itemknn9]
    #
    # from caserec.recommenders.item_recommendation.ItemKNN import ItemKNN
    #
    # for i in range(10):
    #     print("\nRunning Model: *** i:{}/9 (0 ~ 9)***".format(i))
    #
    #     ItemKNN(train_file=list_train_file[i],
    #              test_file=list_test_file[i],
    #              train_plus_enriched_file=list_train_plus_enriched[i],
    #              as_binary=False,
    #              output_file=list_ranking_file[i]).compute(as_table=True)
    #     print("\n")

    train_file0 = 'datasets/filmtrust/train.dat'
    test_file0 = 'datasets/filmtrust/test.dat'
    train_plus_enriched0 = 'datasets/filmtrust/train_plus_enriched_only_when_has_intersection.dat'
    ranking_file_itemknn0 = 'datasets/filmtrust/ranking_file_itemknn_v3_inverted_confidence_metric.dat'

    from caserec.recommenders.item_recommendation.ItemKNN import ItemKNN

    print("\nRunning Model")

    ItemKNN(train_file=train_file0,
             test_file=test_file0,
             train_plus_enriched_file=train_plus_enriched0,
             as_binary=False,
             rank_length=100,
             output_file=ranking_file_itemknn0).compute(as_table=True)
    print("\n")

