import pandas as pd


def edit_distance(str1, str2):
    # https://blog.csdn.net/u010897775/article/details/40018889
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


def norm_edit_distance(str1, str2):
    dist = edit_distance(str1, str2)
    return dist / max(len(str1), len(str2))


def avg_edit_distance(strs1, strs2):
    distances = map(lambda pair: norm_edit_distance(*pair), zip(strs1, strs2))
    return pd.Series(distances).mean()


def avg_accuracy(strs1, strs2):
    return pd.Series(pd.Series(strs1) == pd.Series(strs2)).mean()
