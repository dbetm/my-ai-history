from typing import Dict


def sparse_matrix_multiplication(matrix_a, matrix_b):
    """My solution - accepted"""
    a_num_rows = len(matrix_a)
    a_num_cols = len(matrix_a[0])

    b_num_rows = len(matrix_b)
    b_num_cols = len(matrix_b[0])

    """Matrices A and B can only be multiplied together if the number
    of columns in Matrix A is equal to the number of rows in matrix B.
    """
    if a_num_cols != b_num_rows:
        return [[]]

    # In matrix_b check for each column in which rows there values different to zero
    # then use that info to optimize multiplication
    non_zero_values_per_col: Dict[set] = dict()
    for col in range(b_num_cols):
        for row in range(b_num_rows):
            if matrix_b[row][col] != 0:
                if col in non_zero_values_per_col:
                    non_zero_values_per_col[col].add(row)
                else:
                    non_zero_values_per_col[col] = {row}

    num_ops = 0
    # has the number of rows that matrix A has and the number of columns that matrix B has
    ans = list()
    for i in range(a_num_rows):
        row = list()
        for col_idx in range(b_num_cols):
            total = 0
            for jdx in non_zero_values_per_col.get(col_idx, []):
                total += matrix_a[i][jdx] * matrix_b[jdx][col_idx]
                num_ops += 1
            row.append(total)
        ans.append(row)

    print(f"{num_ops=}")

    return ans


def get_sparse_matrix(matrix):
    sparse_version = dict()

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] != 0:
                sparse_version[(i, j)] = matrix[i][j]

    return sparse_version


def sparse_matrix_multiplication2(matrix_a, matrix_b):
    """Proposal"""
    a_num_rows = len(matrix_a)
    a_num_cols = len(matrix_a[0])

    b_num_rows = len(matrix_b)
    b_num_cols = len(matrix_b[0])

    if a_num_cols != b_num_rows:
        return [[]]

    sparse_matrix_a = get_sparse_matrix(matrix_a)
    sparse_matrix_b = get_sparse_matrix(matrix_b)

    # result matrix has shape: a_num_rows * b_num_cols
    ans = [[0] * b_num_cols for _ in range(a_num_rows)]

    num_ops = 0
    for row_a, col_a in sparse_matrix_a.keys():
        for col_b in range(b_num_cols):
            if (col_a, col_b) in sparse_matrix_b:
                ans[row_a][col_b] += sparse_matrix_a[(row_a, col_a)] * sparse_matrix_b[(col_a, col_b)]
                num_ops += 1

    print(f"{num_ops=}")

    return ans



if __name__ == "__main__":
    n = int(input())
    a = list()
    for _ in range(n):
        tmp = list(map(int, input().split(" ")))
        a.append(tmp)
    
    b = list()
    n2 = int(input(""))
    for _ in range(n2):
        tmp = list(map(int, input().split(" ")))
        b.append(tmp)

    print(sparse_matrix_multiplication(a, b))
    print(sparse_matrix_multiplication2(a, b))