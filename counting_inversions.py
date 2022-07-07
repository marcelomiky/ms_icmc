# l1 = [2, 4, 1, 3, 5]
# l2 = [1, 2, 3, 4, 5]


def getInvCount(arr):
    """"
    Inversions in an array. Follow the ascending order!
    This code is contributed by Smitha Dinesh Semwal
    """
    inv_count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                inv_count += 1

    return inv_count


# l1 = [1455, 16118, 6076, 7637, 1343, 14067, 11577, 1210, 7349, 2222]
# l2 = [1455, 6076, 7637, 1343, 1210, 16118, 7349, 11577, 2222, 14067]
# 10 inversions

# l1 = [5, 4, 3, 2, 1]
# l2 = [1, 2, 3, 4, 5]

# l1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# l2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# l1 = [8758, 15353, 12846, 11454, 158, 16850, 734, 4300, 7324, 15622]
# l2 = [15353, 16850, 4300, 15622, 12846, 11454, 734, 8758, 158, 7324]

# Intersection is Zero
# l1 = [1, 2]
# l2 = [5, 6]


def count_inversions(l1, l2):
    """Compare 2 ranked lists counting the number of inversions. Here is O(n^2)"""

    if len(l1) != len(l2):
        print("The length of the two lists must be the same")
        return 0

    # The case that intersection is ZERO
    elif set(l1).intersection(l2) == set():
        return 0

    else:
        # Dumb way. Gets the 1st list, replace the items for numbers from 0 to len(l1) and
        # replace those numbers in the 2nd list, THEN count the inversions :/

        # Replace the items in the 1st list for the ranked numbers, from o to len(l1)
        aux = [None] * len(l2)

        # Replace those numbers in the 2nd list
        for i in l1:
            if i in l2:  # There are some cases that an item from one list doesnt appear in the other
                aux[l2.index(i)] = l1.index(i)

        # print("aux: ", aux)  # [2, 0, 3, 1, 4]

        return getInvCount(aux)

