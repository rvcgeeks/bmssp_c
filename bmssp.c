/*
 * bmssp.c
 *
 * Single-file BMSSP implementation mirroring:
 * "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
 * (arXiv:2504.17033v2)
 *
 * - Block-structured queue (Lemma 3.3)
 * - Algorithms 1 (FindPivots), 2 (BaseCase), 3 (BMSSP)
 * - DOT reader for simple digraph format:
 *     A -> B [label=3];
 *
 * Usage:
 *   bmssp <file.dot> <source_label> [dest_label]
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define INF DBL_MAX

// ---------------------------------------------------------------------------
// Graph types
// ---------------------------------------------------------------------------

/**
 * @struct Edge
 * Directed edge (u->v) with nonnegative weight w.
 */
typedef struct { 
    int v; 
    double w; 
} Edge;

/**
 * @struct EdgeList
 * Adjacency list for a vertex.
 */
typedef struct {
    Edge *edges;
    int count;
    int capacity;
} EdgeList;

/**
 * @struct Graph
 * Adjacency representation. Node labels are stored in labels[].
 */
typedef struct {
    int n;
    EdgeList *adj;
    char **labels;
} Graph;

/**
 * Create a new graph with n vertices.
 */
Graph* new_graph(int n) {
    Graph *g = (Graph*)malloc(sizeof(Graph));
    g->n = n;
    g->adj = (EdgeList*)calloc(n, sizeof(EdgeList));
    g->labels = (char**)calloc(n, sizeof(char*));
    return g;
}

/**
 * Add directed edge u->v with weight w.
 * Resizes adjacency lists as necessary.
 */
void add_edge(Graph *g, int u, int v, double w) {
    EdgeList *list = &g->adj[u];
    if (list->count == list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 4;
        list->edges = (Edge*)realloc(list->edges, list->capacity * sizeof(Edge));
    }
    list->edges[list->count++] = (Edge){ v, w };
    /* ensure adjacency for v exists */
    if (!g->adj[v].edges && g->adj[v].count == 0) {
        g->adj[v].edges = NULL;
        g->adj[v].capacity = 0;
    }
}

// ---------------------------------------------------------------------------
// Name map for DOT parsing
// ---------------------------------------------------------------------------

typedef struct {
    char name[128];
    int id;
} NameMap;

/**
 * get_id: lookup or insert a string label into NameMap table.
 * This is used by read_dot to map node labels to integer IDs.
 */
int get_id(const char *s, NameMap **map, int *mapSize, int *mapCap) {
    for (int i = 0; i < *mapSize; ++i) {
        if (strcmp((*map)[i].name, s) == 0) return (*map)[i].id;
    }
    if (*mapSize == *mapCap) {
        *mapCap = (*mapCap ? (*mapCap * 2) : 8);
        *map = (NameMap*)realloc(*map, (*mapCap) * sizeof(NameMap));
    }
    int id = *mapSize;
    strncpy((*map)[*mapSize].name, s, sizeof((*map)[*mapSize].name)-1);
    (*map)[*mapSize].name[sizeof((*map)[*mapSize].name)-1] = '\0';
    (*map)[*mapSize].id = id;
    (*mapSize)++;
    return id;
}

/**
 * Read a DOT file (simple directed edges with numeric label attribute) into Graph.
 *
 * Example dot line:
 *   A -> B [label=3];
 *
 * Two-pass parse:
 *  - First pass: collect all node labels and assign ids.
 *  - Second pass: add edges using assigned ids.
 */
Graph* read_dot(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filepath); exit(1); }

    NameMap *map = NULL; int mapSize = 0, mapCap = 0;
    char line[512], src[128], dst[128];
    double w;

    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "->")) {
            if (sscanf(line, " %127s -> %127s [label=%lf];", src, dst, &w) == 3) {
                char *p;
                if ((p = strchr(src, ';'))) *p = '\0';
                if ((p = strchr(dst, ';'))) *p = '\0';
                get_id(src, &map, &mapSize, &mapCap);
                get_id(dst, &map, &mapSize, &mapCap);
            }
        }
    }
    fclose(fp);

    Graph *G = new_graph(mapSize);
    for (int i = 0; i < mapSize; ++i) G->labels[map[i].id] = strdup(map[i].name);

    fp = fopen(filepath, "r");
    if (!fp) { fprintf(stderr, "Cannot re-open %s\n", filepath); exit(1); }
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "->")) {
            if (sscanf(line, " %127s -> %127s [label=%lf];", src, dst, &w) == 3) {
                char *p;
                if ((p = strchr(src, ';'))) *p = '\0';
                if ((p = strchr(dst, ';'))) *p = '\0';
                int u = -1, v = -1;
                for (int i = 0; i < mapSize; ++i) {
                    if (strcmp(map[i].name, src) == 0) u = map[i].id;
                    if (strcmp(map[i].name, dst) == 0) v = map[i].id;
                }
                if (u >= 0 && v >= 0) add_edge(G, u, v, w);
            }
        }
    }
    fclose(fp);
    free(map);
    return G;
}

// ---------------------------------------------------------------------------
// Path struct returned by run_BMSSP
// ---------------------------------------------------------------------------

/**
 * @struct NodePath
 * Holds distance d_hat and Pred for one vertex.
 */
typedef struct { 
    double d_hat; 
    int Pred; 
} NodePath;

/**
 * @struct Path
 * An array of NodePath structs, length pathSize.
 */
typedef struct {
    NodePath *path;
    int pathSize;
} Path;

// ---------------------------------------------------------------------------
// Helper dynamic arrays and binary heap for BaseCase
// ---------------------------------------------------------------------------

typedef struct { 
    double dist; 
    int node; 
} NodeDist;

typedef struct {
    NodeDist *data;
    int size;
    int capacity;
} MinHeap;

void heap_init(MinHeap *h) { h->data = NULL; h->size = 0; h->capacity = 0; }
void heap_free(MinHeap *h) { free(h->data); h->data = NULL; h->size = h->capacity = 0; }
void heap_swap(NodeDist *a, NodeDist *b) { NodeDist tmp = *a; *a = *b; *b = tmp; }

void heap_push(MinHeap *h, NodeDist nd) {
    if (h->size == h->capacity) {
        h->capacity = h->capacity ? h->capacity * 2 : 8;
        h->data = (NodeDist*)realloc(h->data, h->capacity * sizeof(NodeDist));
    }
    int i = h->size++;
    h->data[i] = nd;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].dist <= h->data[i].dist) break;
        heap_swap(&h->data[parent], &h->data[i]);
        i = parent;
    }
}

NodeDist heap_pop(MinHeap *h) {
    NodeDist res = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    while (1) {
        int left = 2*i + 1, right = 2*i + 2, smallest = i;
        if (left < h->size && h->data[left].dist < h->data[smallest].dist) smallest = left;
        if (right < h->size && h->data[right].dist < h->data[smallest].dist) smallest = right;
        if (smallest == i) break;
        heap_swap(&h->data[i], &h->data[smallest]);
        i = smallest;
    }
    return res;
}

int heap_empty(MinHeap *h) { return h->size == 0; }

// ---------------------------------------------------------------------------
// Block-structured queue (Lemma 3.3)
// ---------------------------------------------------------------------------

/**
 * @struct Item
 * key/value pair ⟨vertex, tentative_distance⟩
 */
typedef struct { 
    int key; 
    double value; 
} Item;

/**
 * @struct Block
 * A block of items (sorted by value). Each block holds at most M elements.
 */
typedef struct {
    Item *a;
    int sz;
    int cap;
    double blockMin;
    double blockMax;
} Block;

/**
 * @struct DequeBlocks
 * Two sequences of blocks D0 (batch prepends) and D1 (inserts).
 * bestVal keeps minimal value per vertex in this structure.
 *
 * Lemma 3.3: supports Insert, BatchPrepend, Pull with amortized costs
 * described in the paper. The implementation mirrors the block-based
 * design in the Lemma 3.3 sketch.
 */
typedef struct {
    int M;
    double B_upper;
    Block *D0; int D0_sz; int D0_cap;
    Block *D1; int D1_sz; int D1_cap;
    double *bestVal;
    int nVertices;
} DequeBlocks;

/* ensure block array capacity and initialize new entries */
void ensure_blocks(Block **arr, int *sz, int *cap, int want) {
    if (*cap >= want) return;
    int ncap = (*cap ? (*cap * 2) : (want > 4 ? want : 4));
    if (ncap < want) ncap = want;
    *arr = (Block*)realloc(*arr, sizeof(Block) * ncap);
    for (int i = *cap; i < ncap; ++i) {
        (*arr)[i].a = NULL; (*arr)[i].sz = 0; (*arr)[i].cap = 0;
        (*arr)[i].blockMin = DBL_MAX; (*arr)[i].blockMax = DBL_MAX;
    }
    *cap = ncap;
}

/* ensure single block capacity */
void ensure_blockcap(Block *blk, int want) {
    if (blk->cap >= want) return;
    int ncap = blk->cap ? blk->cap * 2 : (want > 8 ? want : 8);
    if (ncap < want) ncap = want;
    blk->a = (Item*)realloc(blk->a, sizeof(Item) * ncap);
    blk->cap = ncap;
}

/* insert item into a block keeping items sorted by value (stable) */
void block_insert_sorted(Block *blk, Item it) {
    ensure_blockcap(blk, blk->sz + 1);
    int i = blk->sz - 1;
    while (i >= 0 && blk->a[i].value > it.value) { blk->a[i + 1] = blk->a[i]; --i; }
    blk->a[i + 1] = it;
    blk->sz++;
    blk->blockMin = (blk->sz ? blk->a[0].value : DBL_MAX);
    blk->blockMax = (blk->sz ? blk->a[blk->sz - 1].value : DBL_MAX);
}

/* split a block into two halves (simple O(sz) split) */
void split_block(Block *blk, Block *outL, Block *outR) {
    int mid = blk->sz / 2;
    outL->a = NULL; outL->sz = 0; outL->cap = 0; outL->blockMin = DBL_MAX; outL->blockMax = DBL_MAX;
    outR->a = NULL; outR->sz = 0; outR->cap = 0; outR->blockMin = DBL_MAX; outR->blockMax = DBL_MAX;
    ensure_blockcap(outL, mid);
    ensure_blockcap(outR, blk->sz - mid);
    for (int i = 0; i < mid; ++i) outL->a[outL->sz++] = blk->a[i];
    for (int i = mid; i < blk->sz; ++i) outR->a[outR->sz++] = blk->a[i];
    outL->blockMin = (outL->sz ? outL->a[0].value : DBL_MAX);
    outL->blockMax = (outL->sz ? outL->a[outL->sz - 1].value : DBL_MAX);
    outR->blockMin = (outR->sz ? outR->a[0].value : DBL_MAX);
    outR->blockMax = (outR->sz ? outR->a[outR->sz - 1].value : DBL_MAX);
}

/**
 * Initialize DequeBlocks (Lemma 3.3 Initialize(M,B)).
 * Set D0 empty and D1 a single empty block with upper bound B.
 */
void D_Initialize(DequeBlocks *D, int M, double B, int nVertices) {
    D->M = M; D->B_upper = B;
    D->D0 = NULL; D->D0_sz = 0; D->D0_cap = 0;
    D->D1 = NULL; D->D1_sz = 0; D->D1_cap = 0;
    ensure_blocks(&D->D1, &D->D1_sz, &D->D1_cap, 1);
    D->D1_sz = 1;
    D->D1[0].a = NULL; D->D1[0].sz = 0; D->D1[0].cap = 0;
    D->D1[0].blockMin = DBL_MAX; D->D1[0].blockMax = B;
    D->nVertices = nVertices;
    D->bestVal = (double*)malloc(sizeof(double) * nVertices);
    for (int i = 0; i < nVertices; ++i) D->bestVal[i] = DBL_MAX;
}

/**
 * Insert a key/value into D (Lemma 3.3 Insert).
 * Insert to D1: find block whose blockMax >= val and insert there; split if overflow.
 */
void D_Insert(DequeBlocks *D, int key, double val) {
    if (val >= D->B_upper) return;
    if (D->bestVal[key] <= val) return;
    D->bestVal[key] = val;

    int L = 0, R = D->D1_sz - 1, pos = D->D1_sz - 1;
    while (L <= R) {
        int mid = (L + R) >> 1;
        if (D->D1[mid].blockMax >= val) { pos = mid; R = mid - 1; }
        else { L = mid + 1; }
    }
    Block *blk = &D->D1[pos];
    block_insert_sorted(blk, (Item){ key, val });

    if (blk->sz > D->M) {
        Block Lb, Rb;
        split_block(blk, &Lb, &Rb);
        free(blk->a);
        *blk = Lb;
        ensure_blocks(&D->D1, &D->D1_sz, &D->D1_cap, D->D1_sz + 1);
        for (int i = D->D1_sz; i > pos + 1; --i) D->D1[i] = D->D1[i - 1];
        D->D1[pos + 1] = Rb;
        D->D1_sz++;
    }
}

/**
 * Batch prepend items Lset of size Lsz into D0.
 * Lemma 3.3 (BatchPrepend): create blocks at beginning of D0 each ≤ M.
 */
void D_BatchPrepend(DequeBlocks *D, Item *Lset, int Lsz) {
    if (Lsz <= 0) return;
    int idx = 0;
    while (idx < Lsz) {
        int take = D->M;
        if (idx + take > Lsz) take = Lsz - idx;
        Block nb; nb.a = NULL; nb.sz = 0; nb.cap = 0; nb.blockMin = DBL_MAX; nb.blockMax = DBL_MAX;
        for (int i = 0; i < take; ++i) {
            int v = Lset[idx + i].key;
            double val = Lset[idx + i].value;
            if (D->bestVal[v] <= val) continue;
            D->bestVal[v] = val;
            block_insert_sorted(&nb, (Item){ v, val });
        }
        if (nb.sz > 0) {
            ensure_blocks(&D->D0, &D->D0_sz, &D->D0_cap, D->D0_sz + 1);
            for (int i = D->D0_sz; i > 0; --i) D->D0[i] = D->D0[i - 1];
            D->D0[0] = nb;
            D->D0_sz++;
        } else {
            free(nb.a);
        }
        idx += take;
    }
}

/**
 * D_is_empty: return whether D contains no items.
 */
int D_is_empty(const DequeBlocks *D) {
    if (D->D0_sz == 0 && D->D1_sz == 1 && D->D1[0].sz == 0) return 1;
    return 0;
}

/**
 * D_Pull: collect up to M smallest items and return them in outS (outSz),
 * and a separating bound outBound. Implements Lemma 3.3 Pull behavior.
 */
void D_Pull(DequeBlocks *D, Item *outS, int *outSz, double *outBound) {
    *outSz = 0;
    if (D_is_empty(D)) { *outBound = D->B_upper; return; }

    int need = D->M;
    Item *buf = (Item*)malloc(sizeof(Item) * (need * 4 + 8));
    int bsz = 0;
    int acc = 0;

    for (int i = 0; i < D->D0_sz && acc < need; ++i) {
        Block *B0 = &D->D0[i];
        for (int j = 0; j < B0->sz && acc < need; ++j) { buf[bsz++] = B0->a[j]; ++acc; }
        if (acc >= need) break;
    }
    for (int i = 0; i < D->D1_sz && acc < need; ++i) {
        Block *B1 = &D->D1[i];
        for (int j = 0; j < B1->sz && acc < need; ++j) { buf[bsz++] = B1->a[j]; ++acc; }
        if (acc >= need) break;
    }

    int total_items = 0;
    for (int i = 0; i < D->D0_sz; ++i) total_items += D->D0[i].sz;
    for (int i = 0; i < D->D1_sz; ++i) total_items += D->D1[i].sz;

    if (total_items <= D->M) {
        *outSz = 0;
        for (int i = 0; i < D->D0_sz; ++i) {
            Block *B0 = &D->D0[i];
            for (int j = 0; j < B0->sz; ++j) outS[(*outSz)++] = B0->a[j];
        }
        for (int i = 0; i < D->D1_sz; ++i) {
            Block *B1 = &D->D1[i];
            for (int j = 0; j < B1->sz; ++j) outS[(*outSz)++] = B1->a[j];
        }
        for (int i = 0; i < D->D0_sz; ++i) { free(D->D0[i].a); D->D0[i].a = NULL; D->D0[i].sz = D->D0[i].cap = 0; D->D0[i].blockMin = D->D0[i].blockMax = DBL_MAX; }
        D->D0_sz = 0;
        for (int i = 0; i < D->D1_sz; ++i) { free(D->D1[i].a); D->D1[i].a = NULL; D->D1[i].sz = D->D1[i].cap = 0; D->D1[i].blockMin = D->D1[i].blockMax = DBL_MAX; }
        D->D1_sz = 1; D->D1[0].blockMax = D->B_upper;
        *outBound = D->B_upper;
        free(buf);
        return;
    }

    for (int i = 1; i < bsz; ++i) {
        Item x = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j].value > x.value) { buf[j + 1] = buf[j]; --j; }
        buf[j + 1] = x;
    }
    int take = (bsz < need ? bsz : need);
    for (int i = 0; i < take; ++i) outS[i] = buf[i];
    *outSz = take;

    for (int i = 0; i < take; ++i) {
        int v = outS[i].key; double val = outS[i].value;
        int done = 0;
        for (int bi = 0; bi < D->D0_sz && !done; ++bi) {
            Block *B0 = &D->D0[bi];
            for (int j = 0; j < B0->sz; ++j) if (B0->a[j].key == v && B0->a[j].value == val) {
                for (int r = j + 1; r < B0->sz; ++r) B0->a[r - 1] = B0->a[r];
                B0->sz--; done = 1;
                B0->blockMin = (B0->sz ? B0->a[0].value : DBL_MAX);
                B0->blockMax = (B0->sz ? B0->a[B0->sz - 1].value : DBL_MAX);
                break;
            }
        }
        for (int bi = 0; bi < D->D1_sz && !done; ++bi) {
            Block *B1 = &D->D1[bi];
            for (int j = 0; j < B1->sz; ++j) if (B1->a[j].key == v && B1->a[j].value == val) {
                for (int r = j + 1; r < B1->sz; ++r) B1->a[r - 1] = B1->a[r];
                B1->sz--; done = 1;
                B1->blockMin = (B1->sz ? B1->a[0].value : DBL_MAX);
                B1->blockMax = (B1->sz ? B1->a[B1->sz - 1].value : DBL_MAX);
                break;
            }
        }
    }

    double minRem = DBL_MAX;
    int found = 0;
    for (int i = 0; i < D->D0_sz; ++i) if (D->D0[i].sz > 0) { if (D->D0[i].a[0].value < minRem) minRem = D->D0[i].a[0].value; found = 1; }
    for (int i = 0; i < D->D1_sz; ++i) if (D->D1[i].sz > 0) { if (D->D1[i].a[0].value < minRem) minRem = D->D1[i].a[0].value; found = 1; }
    *outBound = (found ? minRem : D->B_upper);

    free(buf);
}

// ---------------------------------------------------------------------------
// Algorithms 1,2,3: FindPivots, BaseCase, BMSSP
// All these accept a context of d_hat[] and Pred[] via pointers (no statics).
// ---------------------------------------------------------------------------

/**
 * FindPivots(Graph *G, double B, int *S, int Ssz,
 *            int *P, int *Psz, int *W, int *Wsz, int k,
 *            double *d_hat, int *Pred)
 *
 * Implements Algorithm 1 (Finding Pivots) sketch:
 *  - Relax for k steps from S with bound B, collect W
 *  - If |W| > k|S|, return P = S
 *  - Else construct forest F on equality edges inside W and select pivots P
 *    as roots with subtree size ≥ k. (We approximate by picking roots proportional)
 *
 * Note: this function updates d_hat and Pred when relaxing.
 */
void FindPivots(Graph *G, double B, const int *S, int Ssz,
                int *P, int *Psz, int *W, int *Wsz, int k,
                double *d_hat, int *Pred)
{
    *Wsz = 0;
    for (int i = 0; i < Ssz; ++i) W[(*Wsz)++] = S[i];
    int *inW = (int*)calloc(G->n, sizeof(int));
    for (int i = 0; i < *Wsz; ++i) inW[ W[i] ] = 1;

    int *layer = NULL; int layer_sz = 0;
    if (Ssz > 0) { layer = (int*)malloc(sizeof(int) * Ssz); memcpy(layer, S, sizeof(int) * Ssz); layer_sz = Ssz; }

    for (int step = 1; step <= k; ++step) {
        int *next = (int*)malloc(sizeof(int) * G->n);
        int next_sz = 0;
        for (int i = 0; i < layer_sz; ++i) {
            int u = layer[i];
            EdgeList *lst = &G->adj[u];
            for (int e = 0; e < lst->count; ++e) {
                int v = lst->edges[e].v; double w = lst->edges[e].w;
                double cand = d_hat[u] + w;
                if (cand <= d_hat[v]) {
                    if (cand < B) {
                        if (!inW[v]) { inW[v] = 1; W[(*Wsz)++] = v; }
                        next[next_sz++] = v;
                        if (cand < d_hat[v]) { d_hat[v] = cand; Pred[v] = u; }
                    }
                }
            }
        }
        free(layer);
        layer = next; layer_sz = next_sz;
        if (*Wsz > k * Ssz) {
            *Psz = 0;
            for (int i = 0; i < Ssz; ++i) P[(*Psz)++] = S[i];
            free(layer); free(inW);
            return;
        }
    }

    /* If |W| <= k|S|, pick at most |W|/k pivots from S.
       For simplicity we choose the first upto quota elements of S as P. */
    *Psz = 0;
    int quota = (k == 0 ? 0 : ((*Wsz + k - 1) / k));
    for (int i = 0; i < Ssz && *Psz < quota; ++i) P[(*Psz)++] = S[i];

    free(layer); free(inW);
}

/**
 * BaseCase(G, B, S) (Algorithm 2)
 * S must be singleton {x} and x complete.
 * Run a mini-Dijkstra starting from x bounded by B, find up to k+1 vertices.
 * Returns B' and U (set of vertices).
 *
 * This function updates d_hat[] and Pred[] in place.
 */
void BaseCase(Graph *G, double B, const int *S, int Ssz,
              double *B_out, int *U_out, int *U_out_sz,
              int k, double *d_hat, int *Pred)
{
    *U_out_sz = 0;
    if (Ssz != 1) { *B_out = B; return; }
    int x = S[0];
    MinHeap H; heap_init(&H);
    heap_push(&H, (NodeDist){ d_hat[x], x });
    int *visited = (int*)calloc(G->n, sizeof(int));
    while (!heap_empty(&H) && *U_out_sz < k + 1) {
        NodeDist nd = heap_pop(&H);
        int u = nd.node; double du = nd.dist;
        if (visited[u]) continue;
        visited[u] = 1;
        U_out[(*U_out_sz)++] = u;
        EdgeList *lst = &G->adj[u];
        for (int e = 0; e < lst->count; ++e) {
            int v = lst->edges[e].v; double w = lst->edges[e].w;
            double cand = d_hat[u] + w;
            if (cand <= d_hat[v] && cand < B) {
                if (cand < d_hat[v]) { d_hat[v] = cand; Pred[v] = u; }
                heap_push(&H, (NodeDist){ d_hat[v], v });
            }
        }
    }
    if (*U_out_sz <= k) { *B_out = B; }
    else {
        double mx = -INFINITY;
        for (int i = 0; i < *U_out_sz; ++i) if (d_hat[U_out[i]] > mx) mx = d_hat[U_out[i]];
        *B_out = mx;
        int w = 0;
        for (int i = 0; i < *U_out_sz; ++i) if (d_hat[U_out[i]] < *B_out) U_out[w++] = U_out[i];
        *U_out_sz = w;
    }
    free(visited); heap_free(&H);
}

/**
 * BMSSP(G, l, B, S) (Algorithm 3)
 *
 * Parameters:
 *  - G: graph
 *  - l: level
 *  - B: upper bound
 *  - S: set of vertices (size Ssz)
 *  - returns: B_prime and U set (U_out, U_out_sz)
 *  - k, t: algorithm parameters
 *
 * This function uses D_Initialize / D_Insert / D_BatchPrepend / D_Pull
 * and calls itself recursively. It updates d_hat[] and Pred[] through
 * nested calls and relaxations.
 */
void BMSSP(Graph *G, int l, double B, const int *S, int Ssz,
           double *B_out, int *U_out, int *U_out_sz,
           int k, int t, double *d_hat, int *Pred)
{
    *U_out_sz = 0;
    if (l == 0) { BaseCase(G, B, S, Ssz, B_out, U_out, U_out_sz, k, d_hat, Pred); return; }

    int *P = (int*)malloc(sizeof(int) * Ssz); int Psz = 0;
    int *W = (int*)malloc(sizeof(int) * (G->n)); int Wsz = 0;
    FindPivots(G, B, S, Ssz, P, &Psz, W, &Wsz, k, d_hat, Pred);

    int M = (1 << (l - 1)) * t;
    if (M < 1) M = 1;
    DequeBlocks Dq; D_Initialize(&Dq, M, B, G->n);
    for (int i = 0; i < Psz; ++i) D_Insert(&Dq, P[i], d_hat[P[i]]);

    int i = 0;
    double B0 = (Psz ? d_hat[P[0]] : B);
    for (int j = 1; j < Psz; ++j) if (d_hat[P[j]] < B0) B0 = d_hat[P[j]];
    double B_prime_local = B0;

    *U_out_sz = 0;
    int U_limit = k * (1 << l) * t;
    Item *Si = (Item*)malloc(sizeof(Item) * M);
    int Sisz = 0;

    int *markU = (int*)calloc(G->n, sizeof(int));

    while ((*U_out_sz) < U_limit && !D_is_empty(&Dq)) {
        ++i;
        double Bi = 0.0;
        D_Pull(&Dq, Si, &Sisz, &Bi);
        int *Si_keys = NULL;
        if (Sisz > 0) { Si_keys = (int*)malloc(sizeof(int) * Sisz); for (int q = 0; q < Sisz; ++q) Si_keys[q] = Si[q].key; }

        double B_prime_i = 0.0;
        int *Ui = (int*)malloc(sizeof(int) * G->n); int Uisz = 0;
        BMSSP(G, l - 1, Bi, Si_keys ? Si_keys : (const int*)Si, Sisz, &B_prime_i, Ui, &Uisz, k, t, d_hat, Pred);

        /* Merge Ui into U_out, deduplicated by markU.
           Record the starting index for newly added vertices so we can
           iteratively process them (propagate relaxations) in this phase. */
        int start_new = *U_out_sz;
        for (int q = 0; q < Uisz; ++q) {
            int uu = Ui[q];
            if (!markU[uu]) { markU[uu] = 1; U_out[(*U_out_sz)++] = uu; }
        }
        if (B_prime_i < B_prime_local) B_prime_local = B_prime_i;

        /* K collects items to batch-prepend (those with values in [B'_i, Bi)). */
        int Kcap = 16;
        Item *K = (Item*)malloc(sizeof(Item) * Kcap); int Ksz = 0;

        /* Iteratively process every newly-added vertex in U_out starting from start_new.
           For each processed vertex u we relax outgoing edges; newly discovered v that are
           not yet in U_out are appended to U_out (and thus will be processed in this loop).
           This guarantees transitive propagation in the same phase and prevents losing nodes
           like F,G,H that are multiple hops away from Ui. */
        for (int proc = start_new; proc < *U_out_sz; ++proc) {
            int u = U_out[proc];
            EdgeList *lst = &G->adj[u];
            for (int e = 0; e < lst->count; ++e) {
                int v = lst->edges[e].v; double w = lst->edges[e].w;
                double cand = d_hat[u] + w;
                if (cand <= d_hat[v]) {
                    if (cand < d_hat[v]) { d_hat[v] = cand; Pred[v] = u; }
                    if (cand >= Bi && cand < B) {
                        D_Insert(&Dq, v, cand);
                    } else if (cand >= B_prime_i && cand < Bi) {
                        if (Ksz >= Kcap) { Kcap *= 2; K = (Item*)realloc(K, sizeof(Item) * Kcap); }
                        K[Ksz++] = (Item){ v, cand };
                    }
                    /* If v not yet recorded in U_out for this BMSSP call, append it
                       so its outgoing edges will also be processed in this same phase. */
                    if (!markU[v]) {
                        markU[v] = 1;
                        if (*U_out_sz < G->n) U_out[(*U_out_sz)++] = v;
                    }
                }
            }
        }

        /* Also consider Si elements whose values fall into [B'_i, Bi) */
        for (int q = 0; q < Sisz; ++q) {
            int x = Si[q].key;
            double vx = d_hat[x];
            if (vx >= B_prime_i && vx < Bi) {
                if (Ksz >= Kcap) { Kcap *= 2; K = (Item*)realloc(K, sizeof(Item) * Kcap); }
                K[Ksz++] = (Item){ x, vx };
            }
        }
        if (Ksz > 0) D_BatchPrepend(&Dq, K, Ksz);

        free(K);
        free(Ui);
        if (Si_keys) free(Si_keys);

        if (D_is_empty(&Dq)) break;
        if ((*U_out_sz) > U_limit) { B_prime_local = B_prime_i; break; }
    }

    for (int q = 0; q < Wsz; ++q) {
        int x = W[q];
        if (d_hat[x] < B_prime_local && !markU[x]) { markU[x] = 1; U_out[(*U_out_sz)++] = x; }
    }

    *B_out = (B_prime_local < B ? B_prime_local : B);

    for (int z = 0; z < Dq.D0_sz; ++z) free(Dq.D0[z].a);
    for (int z = 0; z < Dq.D1_sz; ++z) free(Dq.D1[z].a);
    free(Dq.D0); free(Dq.D1); free(Dq.bestVal);
    free(P); free(W); free(Si); free(markU);
}

// ---------------------------------------------------------------------------
// run_BMSSP wrapper (allocates d_hat and Pred, calls BMSSP, returns Path)
// ---------------------------------------------------------------------------

/**
 * run_BMSSP(G, source)
 * Allocates arrays d_hat[] and Pred[], initializes them, chooses parameters
 * k = floor( (log n)^{1/3} ), t = floor( (log n)^{2/3} ), l = ceil(log2(n)/t).
 * Calls BMSSP and returns a Path object containing NodePath {d_hat, Pred}.
 *
 * Note: caller is responsible for freeing Path.path.
 */
Path run_BMSSP(Graph *G, int source) {
    int n = G->n;
    double *d_hat = (double*)malloc(sizeof(double) * n);
    int *Pred = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) { d_hat[i] = INF; Pred[i] = -1; }
    d_hat[source] = 0.0;

    double log2n = log2((double)n);
    if (log2n <= 0.0) log2n = 1.0;
    int k = (int)fmax(1.0, pow(log2n, 1.0/3.0));
    int t = (int)fmax(1.0, pow(log2n, 2.0/3.0));
    int l = (int)ceil(log2n / (double)t);
    if (l < 0) l = 0;

    /* initial S = {source} */
    int S0[1]; S0[0] = source;
    double B_out;
    int *Ubuf = (int*)malloc(sizeof(int) * n);
    int Usz = 0;

    BMSSP(G, l, INF, S0, 1, &B_out, Ubuf, &Usz, k, t, d_hat, Pred);

    Path P;
    P.pathSize = n;
    P.path = (NodePath*)malloc(sizeof(NodePath) * n);
    for (int i = 0; i < n; ++i) { P.path[i].d_hat = d_hat[i]; P.path[i].Pred = Pred[i]; }

    free(d_hat); free(Pred); free(Ubuf);
    return P;
}

// ---------------------------------------------------------------------------
// Utility: print path by following Pred pointers using labels
// ---------------------------------------------------------------------------

/* recursively print path to node v (by labels) */
void print_path_recursive(const Path *P, int v, Graph *G) {
    if (P->path[v].Pred == -1) {
        printf("%s", G->labels[v]);
    } else {
        print_path_recursive(P, P->path[v].Pred, G);
        printf(" -> %s", G->labels[v]);
    }
}

// ---------------------------------------------------------------------------
// Main: CLI + IO + run_BMSSP usage + printing in requested format
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <file.dot> <source_label> [dest_label]\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    const char *source_label = argv[2];
    const char *dest_label = (argc >= 4 ? argv[3] : NULL);

    Graph *G = read_dot(filename);
    int source = -1, dest = -1;
    for (int i = 0; i < G->n; ++i) {
        if (strcmp(G->labels[i], source_label) == 0) source = i;
        if (dest_label && strcmp(G->labels[i], dest_label) == 0) dest = i;
    }
    if (source < 0) { fprintf(stderr, "Source label '%s' not found\n", source_label); return 1; }
    if (dest_label && dest < 0) { fprintf(stderr, "Destination label '%s' not found\n", dest_label); return 1; }

    Path P = run_BMSSP(G, source);

    printf("Source: %s\n\n", source_label);
    if (dest_label) {
        printf("To %s: ", dest_label);
        if (P.path[dest].d_hat == INF) {
            printf("dist=inf  path=unreachable\n");
        } else {
            printf("dist=%g  path=", P.path[dest].d_hat);
            print_path_recursive(&P, dest, G);
            printf("\n");
        }
    } else {
        for (int i = 0; i < G->n; ++i) {
            printf("To %s: ", G->labels[i]);
            if (P.path[i].d_hat == INF) {
                printf("dist=inf  path=unreachable\n");
            } else {
                printf("dist=%g  path=", P.path[i].d_hat);
                print_path_recursive(&P, i, G);
                printf("\n");
            }
        }
    }

    /* free graph */
    for (int i = 0; i < G->n; ++i) free(G->labels[i]);
    free(G->labels);
    for (int i = 0; i < G->n; ++i) free(G->adj[i].edges);
    free(G->adj);
    free(G);

    /* free path */
    free(P.path);

    return 0;
}
