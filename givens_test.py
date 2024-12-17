import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

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

def generate_random_matrix(size):
    return [[random.uniform(-10, 10) for _ in range(size)] for _ in range(size)]

def test_performance(max_size):
    sizes = list(range(2, max_size + 1))  
    times = []

    for size in sizes:
        A = generate_random_matrix(size)
        
        start_time = time.time()
        qr_algorithm(A)
        elapsed_time = time.time() - start_time
        
        times.append(elapsed_time)
        print(f"Размер матрицы: {size}, Время выполнения: {elapsed_time:.6f} секунд")

    plt.figure(figsize=(10, 6))

    coeffs = np.polyfit(sizes, times, deg=3) 
    poly_eqn = np.poly1d(coeffs)

    x_fit = np.linspace(min(sizes), max(sizes), 100)
    y_fit = poly_eqn(x_fit)

    plt.scatter(sizes, times, marker='o', label='Экспериментальные данные')
    plt.plot(x_fit, y_fit, color='red', label=f'Полиномиальная регрессия (степень {len(coeffs) - 1})')
    
    plt.title('Зависимость времени выполнения от размера матрицы')
    plt.xlabel('Размер матрицы (n)')
    plt.ylabel('Время выполнения (секунды)')
    plt.grid()
    
    plt.legend()
    plt.show()

    print("Уравнение полинома регрессии:")
    print(poly_eqn)

def main():
    choice = input("Хотите протестировать время работы алгоритма? (да/нет): ").strip().lower()
    
    if choice == 'да':
        max_size = int(input("Введите максимальный размер матрицы для тестирования: "))
        test_performance(max_size)
    else:
        file_path = input("Введите путь к файлу с матрицей: ")

        try:
            A = read_matrix_from_file(file_path)
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