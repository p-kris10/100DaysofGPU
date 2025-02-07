
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initWith(float num, float *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = num;
    }
}

void addVectorsInto(float *result, float *a, float *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        result[i] = a[i] + b[i];
    }
}

void test(float target, float *array, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (array[i] != target)
        {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main()
{
    const int N = 2 << 26;
    size_t size = N * sizeof(float);

    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    clock_t start, end;
    start = clock();

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    addVectorsInto(c, a, b, N);

    end = clock();
    float time2 = ((float)(end - start)) / CLOCKS_PER_SEC;
    

    test(7, c, N);
    printf("CPU: %f seconds\n", time2);

    

    free(a);
    free(b);
    free(c);

    return 0;
}
