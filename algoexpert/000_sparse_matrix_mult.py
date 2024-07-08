from typing import Dict


def sparse_matrix_multiplication(matrix_a, matrix_b):
    """My solution - accepted"""
    n = len(matrix_a)
    m = len(matrix_a[0])

    n2 = len(matrix_b)
    m2 = len(matrix_b[0])

    """Matrices A and B can only be multiplied together if the number
    of columns in Matrix A is equal to the number of rows in matrix B.
    """
    if m != n2:
        return [[]]

    # In matrix_b check for each column in which rows there values different to zero
    # then use that info to optimize multiplication
    non_zero_values_per_col: Dict[set] = dict()
    for col in range(m2):
        for row in range(n2):
            if matrix_b[row][col] != 0:
                if col in non_zero_values_per_col:
                    non_zero_values_per_col[col].add(row)
                else:
                    non_zero_values_per_col[col] = {row}

    num_ops = 0
    # has the number of rows that matrix A has and the number of columns that matrix B has
    ans = list()
    for i in range(n):
        row = list()
        for col_idx in range(m2):
            total = 0
            for jdx in non_zero_values_per_col.get(col_idx, []):
                total += matrix_a[i][jdx] * matrix_b[jdx][col_idx]
                num_ops += 1
            row.append(total)
        ans.append(row)

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