#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

#define N 2000  // Размер матрицы

#define MODULUS 2147483647
#define MULTIPLIER 48271

// Функция для измерения времени
float elapsed_msecs(struct timeval s, struct timeval f) {
    return (float) (1000.0 * (f.tv_sec - s.tv_sec) + (0.001 * (f.tv_usec - s.tv_usec)));
}

// Генератор случайных чисел
double linear_congruential_gen(int* seed) {
    const int Q = MODULUS / MULTIPLIER;
    const int R = MODULUS % MULTIPLIER;
    int t = MULTIPLIER * (*seed % Q) - R * (*seed / Q);
    if (t > 0)
        *seed = t;
    else
        *seed = t + MODULUS;
    return (double) *seed / MODULUS;
}

// Последовательное ядро ATAX: y = A^T * (A * x)
void kernel_atax_sequential(double *A, double *x, double *y, double *tmp) {
    // tmp = A * x
    for (int i = 0; i < N; i++) {
        tmp[i] = 0.0;
        for (int j = 0; j < N; j++) {
            tmp[i] += A[i * N + j] * x[j];
        }
    }
    
    // y = A^T * tmp
    for (int j = 0; j < N; j++) {
        y[j] = 0.0;
        for (int i = 0; i < N; i++) {
            y[j] += A[i * N + j] * tmp[i];
        }
    }
}

// Параллельное ядро ATAX с OpenMP
void kernel_atax_parallel(double *A, double *x, double *y, double *tmp) {
    // tmp = A * x
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        tmp[i] = 0.0;
        for (int j = 0; j < N; j++) {
            tmp[i] += A[i * N + j] * x[j];
        }
    }
    
    // y = A^T * tmp
    #pragma omp parallel for
    for (int j = 0; j < N; j++) {
        y[j] = 0.0;
        for (int i = 0; i < N; i++) {
            y[j] += A[i * N + j] * tmp[i];
        }
    }
}

int main() {
    double *A = malloc(sizeof(double) * N * N);
    double *x = malloc(sizeof(double) * N);
    double *y_seq = malloc(sizeof(double) * N);
    double *y_par = malloc(sizeof(double) * N);
    double *tmp_seq = malloc(sizeof(double) * N);
    double *tmp_par = malloc(sizeof(double) * N);
    
    time_t seed_time = time(0);
    
    #pragma omp parallel
    {
        int seed = (seed_time + omp_get_thread_num()) % MODULUS;
        
        #pragma omp for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = linear_congruential_gen(&seed) * N;
            }
        }
        
        #pragma omp for
        for (int i = 0; i < N; i++) {
            x[i] = linear_congruential_gen(&seed) * N;
        }
    }
    
    // Запуск последовательного варианта
    struct timeval start_seq, finish_seq;
    gettimeofday(&start_seq, 0);
    kernel_atax_sequential(A, x, y_seq, tmp_seq);
    gettimeofday(&finish_seq, 0);
    float time_seq = elapsed_msecs(start_seq, finish_seq);
    
    // Запуск параллельного варианта
    struct timeval start_par, finish_par;
    gettimeofday(&start_par, 0);
    kernel_atax_parallel(A, x, y_par, tmp_par);
    gettimeofday(&finish_par, 0);
    float time_par = elapsed_msecs(start_par, finish_par);
    
    // Сравнение результатов
    double max_diff = 0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(y_par[i] / y_seq[i] - 1);
        if (diff > max_diff)
            max_diff = diff;
    }
    
    printf("Sequential Time: %f milliseconds\n", time_seq);
    printf("Parallel Time: %f milliseconds\n", time_par);
    printf("Speedup: %.2fx\n", time_seq / time_par);
    printf("Maximal diff is %lf percents\n", max_diff * 100);
    
    free(A);
    free(x);
    free(y_seq);
    free(y_par);
    free(tmp_seq);
    free(tmp_par);
    
    return 0;
}
