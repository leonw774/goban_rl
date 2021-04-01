#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#define BLACK 1
#define EMPTY 0
#define WHITE -1

/* these function need to be called by c code (Cpython) */
extern "C" {
    // int* territoryEval(int size, int d, int* init_grid);
    // int* getUncondLife(int size, int* init_grid);
    double boardEval (int size, int* init_grid, double komi, int d, double c1, double c2, double c3);
};

/* TERRITORY ESTIMATION */
/*
    # Territory Evaluation Algorithm
        Bouzy, B. (2003). Mathematical Morphology Applied to Computer Go. Int. J. Pattern Recognit. Artif. Intell., 17, 257-268.

    ## Defination

    A = set of elements
    Dilation: D(A) = A + neigbors of A
    Erosion: E(A) = A - neighbors of complements of A
    External Boundary: ExtBound(A) = D(A) - A
    Internal Boundary: IntBound(A) = A - E(A)
    Closing: Close(A) = E(D(A))
    Cloasing is safe territory
    Terriorty Potential Evaluation Operator X(e, d) = E^e . D^d
    X(e, d) is the operation that do dilation d times and then erosion e times

    ## Zobrist's Model To Recognize "Influence"
    assign +64/-64 to black/white points, and 0 elsewhere
    for p in every points:
        for n in neighbors of p:
            if n < 0
                p -= 1
            elif n > 0
                p += 1

    The Zobrist model above has similar effect as dilation.
    So defined that operator as Dz

    Then define Ez in an analogous way:
    for p in every points:
        for n in neighbors of p:
            if color[n] != color[p]:
                if p > 0:
                    p -= 1
                elif p < 0:
                    p += 1

    It is figured that if there ia only one stone on board
    the operater X(e, d) must give same result as identity operator
    Thus e = d * (d - 1) + 1

    X gives better result when d = 4 or 5
    The bigger "d" is , the larger the scale of recognization territories is
*/

typedef struct {
    int size;
    int sizeSquare;
    int* grid;
    int* newgrid;
} DEGrid;

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
void dilate(int d, DEGrid *b) {
    int i = 0;
    int *grid = (*b).grid;
    int *newgrid = (*b).newgrid;
    for (; i < d; i++) {
        int j = 0;
        for (; j < (*b).sizeSquare; j++) {
            int r = 0, nb;
            for(; r < 4; r++) {
                nb = get_neighbor(r, j, (*b).size);
                if (nb == -1) continue;
                if (grid[nb] < 0) {
                    newgrid[j]--;
                }
                else if (grid[nb] > 0) {
                    newgrid[j]++;
                }
            }
        }
        memcpy(grid, newgrid, sizeof(int) * (*b).sizeSquare);
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
void erase(int e, DEGrid *b) {
    int i = 0;
    int *grid = (*b).grid;
    int *newgrid = (*b).newgrid;
    for (; i < e; i++) {
        int j = 0;
        for (; j < (*b).sizeSquare; j++) {
            int r = 0, nb;
            if (grid[j] == 0) continue;
            for(; r < 4; r++) {
                if (newgrid[j] == 0) break; // empty point don't need erase
                nb = get_neighbor(r, j, (*b).size);
                if (nb == -1) continue;
                // because grid[j] != 0 and grid[nb] is not same sign as grid[j]
                if (grid[nb] * grid[j] <= 0) {
                    if (newgrid[j] > 0)
                        newgrid[j]--;
                    else { // won't reach here if newgrid == 0
                        newgrid[j]++;
                    }
                }                   
            }
        }
        memcpy(grid, newgrid, sizeof(int) * (*b).sizeSquare);
    }
};

/*
size: board size squared
d: number of times to do dilate
init_grid: int array of length "size"
            with black = 64, empty = 0, white = -64
return int array of length "size"
*/
int* territoryEval(int size, int d, int* init_grid) {
    DEGrid b;
    //printf("terr_eval\n");
    b.size = size;
    b.sizeSquare = size * size;
    b.grid = (int*) malloc(sizeof(int) * size * size);
    b.newgrid = (int*) malloc(sizeof(int) * size * size);
    //printf("malloc done\n");
    memcpy(b.newgrid, init_grid, sizeof(int) * size * size);
    memcpy(b.grid, init_grid, sizeof(int) * size * size);
    //printf("memcpy done\n");
    int e = d * (d - 1); // + 1; // remove add one if you want a influential estimate 
    dilate(d, &b);
    //printf("dilate done\n");
    erase(e, &b);
    //printf("erase done\n");
    int i = 0;
    free(b.newgrid);
    //printf("free newgrid\n");
    for (int i = 0; i < size * size; ++i) {
        if (b.grid[i] > 64) b.grid[i] = 64;
    }
    return b.grid;
}


/* UNCONDITIONAL LIFE ALGORITHM */
/*
    # Benson's Unconditional Life Algorithm
        http://webdocs.cs.ualberta.ca/~games/go/seminar/2002/020717/benson.pdf
        https://senseis.xmp.net/?BensonsAlgorithm

    ## Definition
    --------
    **Enclosed region**
    The determining color stones enclosing the board into region(s)
    A region consist of opposite color stone and empty points
    R(x) = set of regions enclosed by x-stone
    Example:
    a x + o x o +
    x o b o x o c
    o o o o + x +
    + x x x x x +
    o o + x + + x
    + o o x + d +
    o + o x + + +
        'x': determining color stone
        'o': opposite color stone
        '+': empty point
        'a', 'b', 'c', 'd' are four empty point in four diffrent region enclosed by x

    **Stone block**
        two neighboring same color stones are in the same block.
        B(x) = set of x-color blocks

    **Liberty of a block**
        L(b) = liberty of block b

    **Small region**
        A region is *small* if 
        for all its empty members 'e', there exists a block such that
        'e' belongs to the liberty of that block.

    **Healthy region for a block**
        A region is *healthy* to a block if
        all its empty members is a subset of the liberty of that block.
        Denoted as H(r, b)

    **Vital region to a block in a set of blocks**
        A region is *vital* to a block b in a set of blocks X if
        it is healthy for b and all its neighboring blocks are in X.
        Denoted as V(r, b, X)

    **Unconditional Life**
        A set of block X is *unconditionally alive* if
        there are at least two distinct *vital* region to every block in the set.

    ## Proof of the equivelance of unconditional life and safety
    --------
    ### Notations & Defination
    NB(b) = neighboring region of block b
    NB(r) = neighboring block of region r
    E = set of empty points
    Eye: Let region r is healthy to block b. If #(NB(r)) = 1 than R is an eye
    Joiner: Otherwise, R is a joiner
    Safe: a x-block is safe if
            assuming x pass on every turn,
            any opposite stone placement sequence could not capture this block
    Inside liberty: IL of b belongs to B(x) is any liberty of b in any *small* x-region.
    Outside liberty: OL(b) = L(b) - IL(b)

    Lemma 1: Let b belongs to B(x). 
            (1) If NB(b) contains two or more eyes, b is safe.
            (2) If NB(b) contains only one eye and more than one joiner, 
                then at least one block other than b neighboringeach joiner
                must be captured, before b can be captured
    Theorem 1: Let X be a subset of B(x). If X is *unconditionally alive*,
                then every block in X is safe
    Lemma 2: Let b belongs to B(x) and IL(b) is empty set.
                There exists a legal stones placement sequence where
                opposite color stone place on every point in OL(b) 
    Theorem 2: Let X be the set of all *safe* x-blocks. X is *unconditionally alive*.
    
    ## Algorithm
    --------
    Let:
    X be set of x-block on board
    R(x) be set of small x-regions
    Perform following steps repeatly:
    - Remove from X all x-block with less than 2 *healthy* regions
    - Remove from R all x-region if any of its neighboring block is not in X
    - Stop the algorithm if either steps fails to remove any item
    The remaining block in X are all unconditional alive

    ## Proof of the algorithm
    --------
    Consider the collection of all unconditionally alive blocks sets ordered by subset relation.
    This partially ordered set in a join-semilattice (upper-semilattice)
    This PO set is not lattice, since intersection of two set may not be unconditionally alive

    Blocks b1 and b2 are *joined* if there is a small region r such that
    {b1, b2} is subset of NB(r).
    The transitive closure of this relation is an eqivalence relation:
        b_1 === b_n, if b_1, ..., b_n in B(x) such that b_i and b_i+1 joined, or b_1 = b_n
    denoted as [b_1]

    Let Y be a subset of B(x). [Y] = union([b] for b in Y).
    Consider largest unconditionally alive set X such that X is a subset of Y.
    We call X to be the *support* of Y, denoted as S(Y). Every safe block in Y is in S(Y).

    Let Y be a subset of B(x). Given that small x-region and x-blocks have been determined,
    we can start with only one member in Y = {b}.
    We first compute [Y] by transitive closure algorithm.
    Then to find its support:
    Let Z_0 = [Y], R_0 = { r | NB(r) is in Y and r is small x-region }
    for i >= 0, let Z_i+1 be the set of all blocks in Z_i such that
    they has two healthy regions in R_i.
        Z_i+1 = { b in Z_i | Exist r1, r2 in R_i, r1 != r2, H(r1, b) and H(r2, b) }
    Further, let R_i+1 be the of all small x-regions neighboring Z_i+1.
        R_i+1 = { r | NB(r) is subset of Z_i+1, r is small x-region }
    Let Z be the minimal fixed point of the sequence, that Z = Zn when is Z_n = Z_n+1

    Theorem 3: Z is the largest unconditionally alive set of x-block contained in [Y].
    Proof: Let b in Z. Then exist r1, r2, r1 != r2 such that H(r1,b) and H(r2,b).
        rl, r2 belongs to { r | NB(r) is in Z} to r1 and r2 are vital region to b.
        Hence Z is uncnditionally alive.
        Consider b in [Y] - Z. Now b is in Z_0 but for some i, 0 <= i < n,
        b is in Z and b not in Z_i+l. Therefore, b does not have two healthy regions in R_i.
        Hence b is not unconditionally alive. QED.
*/

class Board {
public:
    int size;
    int sizeSquare;
    int* grid;
    std::vector<std::vector<int>> groups;
    std::vector<int> point2GroupId;

    int get_neighbor(int r, int pos) {
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

    Board(int size, int* init_grid):
    size(size), sizeSquare(size*size) {
        grid = new int[sizeSquare];
        memcpy(grid, init_grid, sizeof(int) * sizeSquare);
        point2GroupId.resize(sizeSquare, -1);
        build_groups();
    }

    void build_groups() {
        for (int i = 0; i < sizeSquare; ++i) {
            if (grid[i] == EMPTY) continue;
            if (point2GroupId[i] != -1) continue; // already built
            int gcolor = grid[i];
            std::vector<int> expander;
            std::vector<int> searched;
            expander.push_back(i);
            
            while (!expander.empty()) {
                int p = expander.back();
                expander.pop_back();
                searched.push_back(p);
                for (int r = 0; r < 4; ++r) {
                    int nb = get_neighbor(r, p);
                    if (nb == -1) continue;
                    if (grid[nb] == gcolor &&
                        std::count(searched.begin(), searched.end(), nb) == 0)
                    {
                        expander.push_back(nb);
                    }
                }
            }
            groups.push_back(searched); // add new group
            // update point2GroupId
            for (int p: groups.back()) {
                point2GroupId[p] = groups.size() - 1;
            }
        }
    }

    int* getUncondLiveGrid() {
        int* result_grid = new int[sizeSquare];
        memset(result_grid, 0, sizeof(int) * sizeSquare);

        for (int color = -1; color <= 1; color += 2) {
            // std::cout << "color" << color << "\n";
            // {rid : points}
            // use map because erasion is constant
            std::unordered_map<int, std::unordered_set<int>> smallRegions;
            // {rid : gids}
            // use vector because push_back is constant
            std::vector<std::unordered_set<int>> rid2NBGid;
            // gids that neighboring small regions
            std::unordered_set<int> nbGid; 

            // find small regions and thier neighbor blocks
            std::unordered_set<int> all_searched;
            for (int i = 0; i < sizeSquare; ++i) {
                if (grid[i] == color) continue;
                if (all_searched.count(i)) continue;
                bool is_small = true;
                std::vector<int> expander;
                std::unordered_set<int> searched;
                std::unordered_set<int> tmpNBGroupIds; // member is gid
                expander.push_back(i);
                
                while (expander.size() > 0) {
                    int p = expander.back();
                    expander.pop_back();
                    bool hasNBStones = false;
                    for (int r = 0; r < 4; ++r) {
                        int nb = get_neighbor(r, p);
                        if (nb == -1) continue;
                        if (grid[nb] == color) {
                            hasNBStones = true;
                            tmpNBGroupIds.insert(point2GroupId[nb]);
                        }
                        else if (searched.count(nb) == 0) {
                            expander.push_back(nb);
                        }
                    }
                    is_small = is_small && hasNBStones;
                    searched.insert(p);
                }
                all_searched.insert(searched.begin(), searched.end());
                if (is_small) {
                    int rid = rid2NBGid.size();
                    smallRegions.insert({{rid, searched}});
                    rid2NBGid.push_back(tmpNBGroupIds);
                    nbGid.insert(tmpNBGroupIds.begin(), tmpNBGroupIds.end());
                }
            }
            
            // std::cout << "smallR: ";
            // for (auto it = smallRegions.begin(); it != smallRegions.end(); ++it) {
            //     std::cout << "rid:" << it->first << " : ";
            //     for (int y: it->second) {
            //         std::cout << y << " ";
            //     }
            //     std::cout << ",";
            // }
            // std::cout << "\n";
            
            if (smallRegions.size() < 2) continue;
            // end for: small regions and thier neighbor blocks
            // find healthy relationship of blocks to regions
            std::unordered_multimap<int, int> H; // <gid, rid>
            for (auto it = smallRegions.begin(); it != smallRegions.end(); ++it) {
                int key = it->first;
                std::vector<int> emptyPoints;
                for (auto setit = it->second.begin(); setit != it->second.end(); ++setit) {
                    if (grid[*setit] == EMPTY) {
                        emptyPoints.push_back(*setit);
                    }
                }
                // test if every nbgroup if it is neighboring all empty point
                for (int gid: rid2NBGid[key]) {
                    bool isHealthy = std::all_of(
                        emptyPoints.begin(),
                        emptyPoints.end(),
                        [this, gid](int ep) -> bool {
                            for (int r = 0; r < 4; ++r) {
                                int nb = get_neighbor(r, ep);
                                if (nb == -1) continue;
                                if (point2GroupId[nb] == gid)
                                    return true;
                            }
                            return false;
                        }
                    );
                    if (isHealthy) { // region i is healthy to group j 
                        H.insert(std::pair<int, int>(gid, key));
                    }
                }
            }
            if (H.size() < 2) continue;
            // std::cout << "Healthy\n";
            // for (auto &p: H) {
            //     std::cout << "gid:" << p.first << " : " << p.second << ", ";
            // }
            // std::cout << "\n";
            
            // end for: healthy relationship of blocks to regions
            // do the algorithm iteration:
            // - Remove from X all x-block with less than 2 *healthy* regions
            // - Remove from R all x-region if any of its neighboring block is not in X
            // - Stop the algorithm if either steps fails to remove any item
            while (true) {
                for (auto it = nbGid.begin(); it != nbGid.end(); /* empty */) {
                    if (H.count(*it) < 2) {
                        it = nbGid.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                
                // std::cout << "GID\n";
                // for (int p: nbGid) {
                //     std::cout << p << ", ";
                // }
                // std::cout << "\n";
               

                int prev_size = smallRegions.size();
                for (auto it = smallRegions.begin(); it != smallRegions.end(); /* empty */) {
                    bool remove = std::any_of(
                        rid2NBGid[(*it).first].begin(), rid2NBGid[(*it).first].end(), 
                        [nbGid](int gid) -> bool {
                            return !nbGid.count(gid);
                        });
                    if (remove) {
                        it = smallRegions.erase(it);
                    }
                    else {
                        ++it;
                    }
                }
                
                // std::cout << "smallR\n";
                // for (auto &e: smallRegions) {
                //     std::cout << "rid " << e.first << " : ";
                //     for (int p: e.second) {
                //         std::cout << p << ",";
                //     }
                // }
                // std::cout << "\n";
               

                if (prev_size == smallRegions.size()) {
                    break;
                }
                else {
                    // update H
                    for (auto it = H.begin(); it != H.end(); /* empty */) {
                        if (smallRegions.count(it->second)) {
                            ++it;
                        }
                        else {
                            it = H.erase(it);
                        }
                    }
                }
            }
            // end while: algorithm iteration
            // build result grid from nbGid & smallRegions
            for (int gid: nbGid) {
                for (int point: groups[gid]) {
                    result_grid[point] = color;
                }
                for (auto &pair: smallRegions) {
                    for (int point: pair.second) {
                        result_grid[point] = color;
                    }
                }
            }
        }
        return result_grid;
    }

    // lib count of each group
    std::vector<std::unordered_set<int>> gidLib;

    int* getAvgLiberty() {
        int* avgLib = new int[2];
        int bgcount = 0, wgcount = 0;
        avgLib[0] = 0; avgLib[1] = 0;
        gidLib.resize(groups.size());
        for (int i = 0; i < groups.size(); ++i) {
            for (int p: groups[i]) {
                for (int r = 0; r < 4; ++r) {
                    int nb = get_neighbor(r, p);
                    if (nb == -1) continue;
                    if (grid[nb] == 0) gidLib[i].insert(nb);
                }
            }
            if (grid[groups[i][0]] == BLACK) {
                avgLib[0] += gidLib[i].size() * groups[i].size();
                bgcount++;
            }
            else {
                avgLib[1] += gidLib[i].size() * groups[i].size();
                wgcount++;
            }
        }
        if (bgcount > 0) avgLib[0] /= bgcount;
        if (wgcount > 0) avgLib[1] /= wgcount;
        return avgLib; 
    }
};

/* 
    return size * size grid of unconditionally alive stones and thier inner territory
    with black = 1, empty = 0, white = -1
*/
int* getUncondLife(int size, int* init_grid) {
    Board b(size, init_grid);
    int* uncond_live_grid = b.getUncondLiveGrid();
    return uncond_live_grid;
}

/* LIBERTY RACE */
/*
    this calculates the liberty race situation of each locals
    a local is a set of groups that neighboring each other

    for each locals:
        calculate liberty of each groups
        find critical groups from both colors:
            if a group's liberty count is less than all neighboring opponent group's liberty count
            then that group is a critical group
        
        if all have no critical group: tie
        elif one have critical group: the one that have lose
        else:
            both color find the critical groups of minimum liberties
            the one with more liberties win
            if tie:
                the one with more minimum liberties groups win
                if still tie: then its tie
        the color that wins can get points of the amount of enemy stones in the local
    return final score of black & white in array: {Black score, White score}
*/
/* 
int* getLibertyRaceScore(Board& builtBoard) {

}
*/

/* POLICY FUNCTION */
/*
    the smaller abs value of estimated territory, the higher policy value on that point
    the policy value around low liberty groups are also higher
    in every point i:
        terr_score[i] = (64 - terr_grid[i]) * (4 + terr_grid[i]) / 1156
        libt_score[i] = exp(-min(libt of nb groups) + 1)
            --when min of nb group's liberty is one, libt_score = exp(0) = 1
    policy = terr_score * ct + libt_score * cl
    pass (id: size*size) is mean of non-zero policy
*/
double* calcPolicy (Board b, int* terr_grid, double ct, double cl) {
    double* result_grid = new double[b.sizeSquare+1];
    memset(result_grid, 0.0, sizeof(double) * b.sizeSquare+1);
    
    double sumnonzero = 0.0;
    int countnonzero = 0;
    for (int i = 0; i < b.sizeSquare; ++i) {
        if (b.grid[i] != 0) continue;
        int minNBLib = 10000;
        for (int r = 0; r < 4; ++r) {
            int nb = b.get_neighbor(r, i);
            if (nb == -1) continue;
            if (b.point2GroupId[nb] == -1) continue;
            size_t libsize = b.gidLib[b.point2GroupId[nb]].size();
            if (libsize < minNBLib) minNBLib = libsize;
        }
        double terrScore = (64 - terr_grid[i]) * (4 + terr_grid[i]) / 1156.0;
        if (terrScore < 0) terrScore = 0;
        result_grid[i] = ct * terrScore + 
                         cl * ((minNBLib < 9999) ? exp(1 - minNBLib) : 0);
        if (result_grid[i] > 0) {
            sumnonzero += result_grid[i];
            countnonzero++;
        }
        // std::cout << terrScore << ", " << minNBLib << ", " << result_grid[i] << std::endl;
    }
    result_grid[b.sizeSquare] = sumnonzero / countnonzero;
    return result_grid;
}

/* THE ONE FUNCTION TO DO IT ALL */
/*
    PARAMETERS:
    d is the parameter for territory estimation
    init_grid is original board grid with black = 1, empty = 0, white = -1
    c1, c2, c3: parameters for
        c1: estimated territory + 
        c2: liberty race situation + 
        c3: territory based on only unconditionally alive groups

    RETURN:
    array of float with length (size * size + 2)
    index 0 ~ size*size is policy grid
        policy:
            see that comment at calcPolicy()
    index size*size+1 is evaluation value
        evaluation value = black score - white score
*/
double boardEval (int size, int* init_grid, int d, double c1, double c2, double c3) {
    int* terr_init_grid = new int[size * size];
    for (int i = 0; i < size * size; ++i) {
        terr_init_grid[i] = init_grid[i] * 64;
    }
    int* terr_result_grid = territoryEval(size, d, terr_init_grid);
    Board b(size, init_grid);
    int* uncond_life_result_grid = b.getUncondLiveGrid();
    int* avgLiberty = b.getAvgLiberty();
    // int* liberty_race_score = getLibertyRaceScore(b);
    //double* policy_grid = calcPolicy(b, terr_result_grid, 0.25, 0.75);
    // std::cout << "A";
    // calc final eval score
    double bscore = 0, wscore = 0; //komi * 0.5;
    for (int i = 0; i < size * size; ++i) {
        if (terr_result_grid[i] > 0) bscore += c1;
        else if (terr_result_grid[i] < 0) wscore += c1;
    }
    for (int i = 0; i < size * size; ++i) {
        if (uncond_life_result_grid[i] > 0) bscore += c2;
        else if (uncond_life_result_grid[i] < 0) wscore += c2;
    }
    bscore += avgLiberty[0] * c3;
    wscore += avgLiberty[1] * c3;
    // std::cout << "B";
    // double* finalarray = new double[size*size+2];
    // memcpy(finalarray, policy_grid, sizeof(double) * (size*size+1));
    // finalarray[size*size+1] = bscore - wscore;
    return bscore - wscore;
}

