import math

def read_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            matrix.append(row)
    return matrix

def multiply_matrices(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Матрицы не могут быть перемножены.")
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    
    return result

def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)
    
    return det

def givens_qr(A):
    m, n = len(A), len(A[0])
    Q = [[float(i == j) for j in range(n)] for i in range(m)] 
    R = [row[:] for row in A]
    
    for k in range(n):
        for i in range(m-1, k, -1):  
            if abs(R[i][k]) > 1e-15:  
                r = math.sqrt(R[i-1][k]**2 + R[i][k]**2)
                c = R[i-1][k] / r
                s = -R[i][k] / r
                for j in range(k, n):
                    temp1 = c * R[i-1][j] - s * R[i][j]
                    temp2 = s * R[i-1][j] + c * R[i][j]
                    R[i-1][j] = temp1
                    R[i][j] = temp2
                for j in range(m):
                    temp1 = c * Q[j][i-1] - s * Q[j][i]
                    temp2 = s * Q[j][i-1] + c * Q[j][i]
                    Q[j][i-1] = temp1
                    Q[j][i] = temp2
    
    return Q, R

def qr_algorithm(A, epsilon=1e-15, max_iterations=10000):
    n = len(A)
    A_k = [row[:] for row in A]
    eigenvalues_old = [0] * n
    Q_total = [[float(i == j) for j in range(n)] for i in range(n)]  

    for iteration in range(max_iterations):
        Q, R = givens_qr(A_k)
        A_k = multiply_matrices(R, Q)

        Q_total = multiply_matrices(Q_total, Q)

        eigenvalues_new = [A_k[i][i] for i in range(n)]

        if all(abs(eigenvalues_new[i] - eigenvalues_old[i]) < epsilon for i in range(n)):
            break
        
        eigenvalues_old = eigenvalues_new[:]
    
    eigenvalues = [A_k[i][i] for i in range(n)]
    eigenvectors = [Q_total[i] for i in range(n)]
    
    return eigenvalues, eigenvectors

def main():
    file_path = input("Введите путь к файлу с матрицей: ")

    try:
        A = read_matrix_from_file(file_path)

        if len(A) != len(A[0]):
            raise ValueError("Матрица должна быть квадратной.")

        det_A = determinant(A)
        print(f"Определитель матрицы: {det_A}")
        
        if abs(det_A) < 1e-15: 
            print("Матрица вырождена. Программа завершает работу.")
            return
            
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    eigenvalues, eigenvectors = qr_algorithm(A)

    print("\nСобственные значения:")
    for val in eigenvalues:
        print(val)

    print("\nСобственные векторы:")
    for vec in eigenvectors:
        print(vec)

if __name__ == "__main__":
    main()