#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
    Q[not children] = -inf
    U = curnode.Q + np.nan_to_num(self.c_put * np.sqrt(np.log(np.sum(curnode.N)) / curnode.N), posinf)
*/
double* uct(double* Q, int* N, double cput, double posinf, unsigned int size) {
    double* U = malloc(sizeof(double) * size);
    memcpy(U, Q, sizeof(double) * size);
    int i;
    double logsumN = 0;
    for (i = 0; i < size; i++) logsumN += N[i];
    logsumN = log(logsumN);
    for (i = 0; i < size; i++) {
        if (N[i] == 0)
            U[i] += posinf;
        else 
            U[i] += sqrt(logsumN / (double)N[i]);
    }
    return U;
}