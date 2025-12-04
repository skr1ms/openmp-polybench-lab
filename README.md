# ЛР2

Скачать набор тестов polybench-c-4.2.1-beta.tar.gz, прочитать документацию polybench.pdf и выбрать алгоритм для распараллеливания. Распараллелить с помощью директив OpenMP (пример директивы, которая распараллеливает цикл for: `#pragma omp parallel for`).

**Запрещенный алгоритм:** произведение матриц 2mm.

**Пример-болванка:** fp.tar.gz

## Методическая рекомендация

- изъять реализацию ядра (kernel) из исходных текстов Polybench;
- разработать обвязку самостоятельно:
  - инициализация входных данных;
  - запуск последовательного варианта;
  - запуск параллельного варианта;
  - сравнение результатов последовательного и параллельного вариантов;
  - сравнение времени выполнения последовательного и параллельного вариантов.

## Как измерять время выполнения?

```c
#include <time.h>
#include <sys/time.h>

…

float elapsed_msecs(struct timeval s, struct timeval f) {
    return (float) (1000.0 * (f.tv_sec - s.tv_sec) + (0.001 * (f.tv_usec - s.tv_usec)));
}

…

struct timeval start_time;
gettimeofday(&start_time, 0);

…your kernel…

struct timeval finish_time;
gettimeofday(&finish_time, 0);

float elapsedTime = elapsed_msecs(start_time, finish_time);
printf("Elapsed Time: %f milliseconds\n", elapsedTime);
```

## Как сравнить результаты двух вариантов решения?

```c
#include <math.h>

…

// Пусть Cv и С - две матрицы размера N*N.
// Каждая хранит результат одного варианта решения.
// Матрицы линеаризованы по строкам.

double max_diff = 0;
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        double diff = fabs(Cv[i * N + j] / C[i * N + j] - 1);
        if (diff > max_diff)
            max_diff = diff;
    }
}

printf("Maximal diff is %lf percents\n", max_diff * 100);
```

## Как инициализировать массивы?

```c
#include <omp.h>

…

// Uniform random function code is based on linear congruential generator
// http://www.cs.wm.edu/~va/software/park/park.html

#define MODULUS 2147483647
#define MULTIPLIER 48271

/**
 * Returns a pseudo-random real number uniformly distributed between 0 and 1.
 */
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

…

// Пусть A и B - две матрицы размера N*N.
// Они будут заполнены одними и теми же случайными значениями.
// Матрицы линеаризованы по строкам.

time_t seed_time = time(0);

#pragma omp parallel
{
    int seed = (seed_time + omp_get_thread_num()) % MODULUS;
    #pragma omp for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i * N + j] = linear_congruential_gen(&seed) * N * N;
            B[i * N + j] = A[i * N + j];
        }
}
```

## Как выделить память?

```c
// Пусть A - матрица размера N*N с элементами типа double.

#include <stdlib.h>

…

double * A = malloc(sizeof(double) * N * N);
…your code…
free(A);
```

## Как собирать код?

```bash
gcc -std=c99 -fopenmp program.c -o program
```

