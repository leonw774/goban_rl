#include <stdlib.h>
#include <math.h>

/*
    Q[not children] = -inf
    U = curnode.Q + np.nan_to_num(self.c_put * np.sqrt(np.log(np.sum(curnode.N)) / curnode.N), posinf=2.0)
*/
int* uct(int* Q, int*N, double cput, double posinf, size_t size) {
    double* U = malloc(sizeof(double) * size);
    int i;
    double sumN = 0, logsumN;
    for (i = 0; i < size; i++) sumN += N[i];
    logsumN = log(sumN); 
    for (i = 0; i < size; i++) {
        if (N[i] == 0)
            U[i] = posinf;
        else 
            U[i] = Q[i] + sqrt(logsumN / (double)N[i]);
    }
    return U;
}