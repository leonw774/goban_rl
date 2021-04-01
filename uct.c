#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
    Q[valid] = [-1, 1]
    Q[invalid] = -inf
    U = Q + c_put * Policy * (sqrt(sum(N) / (N + 1))
*/
double* uct(double* Q, int* N, double cput, unsigned int size) {
    double* U = (double*) malloc(sizeof(double) * size);
    memcpy(U, Q, sizeof(double) * size);
    int i;
    double logsumN = 0;
    for (i = 0; i < size; i++) logsumN += N[i];
    logsumN = log(logsumN);
    for (i = 0; i < size; i++) {
        if (N[i] == 0) U[i] += cput; // to balance exploitation & exploration 
        else U[i] += cput * sqrt(logsumN / (double) N[i]);
    }
    return U;
}