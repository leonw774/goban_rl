#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Board {
    int size;
    int size_square;
    int* grid;
    int* new_grid;
} Board;

int get_neighbor(int r, int pos, int size) {
    switch(r) {
        case 0: // top
            if (pos >= size)
                return pos - size;
            break;
        case 1: // left
            if (pos % size != 0)
                return pos - 1;
            break;
        case 2: // down
            if (pos < size * (size - 1))
                return pos + size;
            break;
        case 3: // right
            if (pos % size != size - 1) 
                return pos + 1;
            break;
    }
    return -1;
}
/*
for p in every points:
    for n in neighbors of p:
        if n < 0
            p -= 1
        elif n > 0
            p += 1
*/
void dilate(int d, Board *b) {
    int i = 0;
    int *grid = (*b).grid;
    int *new_grid = (*b).new_grid;
    for (; i < d; i++) {
        int j = 0;
        for (; j < (*b).size_square; j++) {
            int r = 0, nb;
            for(; r < 4; r++) {
                nb = get_neighbor(r, j, (*b).size);
                if (nb == -1) continue;
                if (grid[nb] < 0) {
                    new_grid[j]--;
                }
                else if (grid[nb] > 0) {
                    new_grid[j]++;
                }
            }
        }
        memcpy(grid, new_grid, sizeof(int) * (*b).size_square);
    }
};

/*
for p in every points:
    for n in neighbors of p:
        if color[n] != color[p] and color[p] != empty:
            if p > 0:
                p -= 1
            elif p < 0:
                p += 1
*/
void erase(int e, Board *b) {
    int i = 0;
    int *grid = (*b).grid;
    int *new_grid = (*b).new_grid;
    for (; i < e; i++) {
        int j = 0;
        for (; j < (*b).size_square; j++) {
            int r = 0, nb;
            if (grid[j] == 0) continue;
            for(; r < 4; r++) {
                if (new_grid[j] == 0) break; // empty point don't need erase
                nb = get_neighbor(r, j, (*b).size);
                if (nb == -1) continue;
                // because grid[j] != 0 and grid[nb] is not same sign as grid[j]
                if (grid[nb] * grid[j] <= 0) {
                    if (new_grid[j] > 0)
                        new_grid[j]--;
                    else { // won't reach here if new_grid == 0
                        new_grid[j]++;
                    }
                }                   
            }
        }
        memcpy(grid, new_grid, sizeof(int) * (*b).size_square);
    }
};

/*
size: board size squared
d: number of times to do dilate
init_grid: int array of length "size"
return int array of length "size"
*/
int* _board_eval(int size, int d, int* init_grid) {
    Board b;
    //printf("_board_eval\n");
    b.size = size;
    b.size_square = size * size;
    b.grid = init_grid;
    b.new_grid = malloc(sizeof(int) * size * size);
    //printf("malloc done\n");
    memcpy(b.new_grid, init_grid, sizeof(int) * size * size);
    //printf("memcpy done\n");
    int e = d * (d - 1) + 1;
    dilate(d, &b);
    //printf("dilate done\n");
    erase(e, &b);
    //printf("erase done\n");
    int i = 0;
    free(b.new_grid);
    //printf("free new_grid\n");
    return b.grid;
}