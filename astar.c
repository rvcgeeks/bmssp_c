
#include "graph.h"

/* --------------------------- A* search --------------------------- */

/**
 * @struct PQItem
 * @brief Item stored in the A* priority queue with f = g+h priority.
 */
typedef struct {
    int node;
    double f;
    double g;
} PQItem;

/**
 * @struct MinHeap
 * @brief Binary min-heap over PQItem based on f-value.
 */
typedef struct {
    PQItem *data;
    int size;
    int cap;
} MinHeap;

/**
 * @brief Initialize heap.
 */
void heap_init(MinHeap *h) {
    h->data = NULL;
    h->size = 0;
    h->cap = 0;
}

/**
 * @brief Free heap internal memory.
 */
void heap_free(MinHeap *h) { free(h->data); }

/**
 * @brief Push item into heap.
 */
void heap_push(MinHeap *h, PQItem it) {
    if (h->size + 1 >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 32;
        h->data = (PQItem*)realloc(h->data, h->cap * sizeof(PQItem));
    }
    int i = ++h->size;
    h->data[i] = it;
    while (i > 1) {
        int p = i / 2;
        if (h->data[p].f <= h->data[i].f) break;
        PQItem tmp = h->data[p]; h->data[p] = h->data[i]; h->data[i] = tmp;
        i = p;
    }
}

/**
 * @brief Pop min item (f) from heap. Caller must ensure non-empty.
 */
PQItem heap_pop(MinHeap *h) {
    PQItem res = h->data[1];
    h->data[1] = h->data[h->size--];
    int i = 1;
    while (1) {
        int l = i * 2, r = l + 1, best = i;
        if (l <= h->size && h->data[l].f < h->data[best].f) best = l;
        if (r <= h->size && h->data[r].f < h->data[best].f) best = r;
        if (best == i) break;
        PQItem tmp = h->data[i]; h->data[i] = h->data[best]; h->data[best] = tmp;
        i = best;
    }
    return res;
}

/**
 * @brief Check if heap is empty.
 */
int heap_empty(MinHeap *h) { return h->size == 0; }

/*
 * Heuristic function h(u) for A* declaration.
 */
double heuristic(int u, int target, const Graph *G);

/**
 * @brief Reconstruct path from start to goal using predecessor array.
 *
 * Returns dynamically allocated array of node ids (path length stored),
 * caller must free returned array.
 */
int* reconstruct_path(int *pred, int start, int goal, int *out_len) {
    if (goal < 0) { *out_len = 0; return NULL; }
    int cap = 16;
    int *rev = (int*)malloc(sizeof(int) * cap);
    int rsz = 0;
    int cur = goal;
    while (cur != -1) {
        if (rsz >= cap) { cap *= 2; rev = (int*)realloc(rev, sizeof(int) * cap); }
        rev[rsz++] = cur;
        if (cur == start) break;
        cur = pred[cur];
    }
    /* if we didn't reach start, path doesn't exist */
    if (rsz == 0 || rev[rsz - 1] != start) {
        free(rev);
        *out_len = 0;
        return NULL;
    }
    /* reverse to produce path from start to goal */
    int *path = (int*)malloc(sizeof(int) * rsz);
    for (int i = 0; i < rsz; ++i) path[i] = rev[rsz - 1 - i];
    *out_len = rsz;
    free(rev);
    return path;
}

/**
 * @brief A* search from source to target on graph G.
 *
 * Returns distance via outDist and predecessor array (dynamically allocated)
 * via outPred. Caller must free outPred. If target unreachable, returns 0 and
 * outDist set to INF and outPred filled with -1s.
 */
int astar(const Graph *G, int source, int target, double *outDist, int **outPred) {
    int n = G->n;
    double *gscore = (double*)malloc(sizeof(double) * n);
    int *pred = (int*)malloc(sizeof(int) * n);
    int *closed = (int*)calloc(n, sizeof(int));

    for (int i = 0; i < n; ++i) { gscore[i] = INF; pred[i] = -1; }
    gscore[source] = 0.0;

    MinHeap open; heap_init(&open);
    double f0 = gscore[source] + heuristic(source, target, G);
    heap_push(&open, (PQItem){ source, f0, gscore[source] });

    while (!heap_empty(&open)) {
        PQItem cur = heap_pop(&open);
        int u = cur.node;
        if (closed[u]) continue; /* lazy removal */
        if (u == target) {
            /* found */
            *outDist = gscore[target];
            *outPred = pred;
            free(gscore); free(closed);
            heap_free(&open);
            return 1;
        }
        closed[u] = 1;
        EdgeList *lst = &G->adj[u];
        for (int ei = 0; ei < lst->count; ++ei) {
            int v = lst->edges[ei].v;
            double w = lst->edges[ei].w;
            double cand = gscore[u] + w;
            if (cand < gscore[v]) {
                gscore[v] = cand;
                pred[v] = u;
                double fv = cand + heuristic(v, target, G);
                heap_push(&open, (PQItem){ v, fv, cand });
            }
        }
    }

    /* not found */
    *outDist = INF;
    *outPred = pred;
    free(gscore); free(closed);
    heap_free(&open);
    return 0;
}

/* --------------------------- Heuristic Function Definition --------------------------- */

/**
 * @brief Heuristic function h(u) for A*.
 *
 * Since DOT input does not include coordinates, we use an admissible
 * heuristic h(u)=0 for all nodes (A* reduces to Dijkstra). Replace this
 * function if you have additional information.
 */
double heuristic(int u, int target, const Graph *G) {
    (void)u; (void)target; (void)G;
    return 0.0;
}

#ifndef NODRIVER

/* --------------------------- Printing --------------------------- */

/**
 * @brief Print path in requested format "A -> B -> C" using labels.
 */
void print_path_labels(const Graph *G, const int *path, int plen) {
    if (plen <= 0) return;
    printf("%s", G->labels[path[0]]);
    for (int i = 1; i < plen; ++i) {
        printf(" -> %s", G->labels[path[i]]);
    }
}

/* ------------------------------- main ------------------------------- */

/**
 * CLI: astar.exe sample.dot A H
 * Destination is mandatory.
 *
 * Output example:
 * Source: A
 *
 * To H: dist=12  path=A -> ... -> H
 */
int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <file.dot> <source> <target>\n", argv[0]);
        return 1;
    }

    const char *file = argv[1];
    const char *srcName = argv[2];
    const char *tgtName = argv[3];

    Graph *G = read_dot(file);

    int source = -1, target = -1;
    for (int i = 0; i < G->n; ++i) {
        if (strcmp(G->labels[i], srcName) == 0) source = i;
        if (strcmp(G->labels[i], tgtName) == 0) target = i;
    }
    if (source < 0) { fprintf(stderr, "Source '%s' not found\n", srcName); return 1; }
    if (target < 0) { fprintf(stderr, "Target '%s' not found\n", tgtName); return 1; }

    printf("Source: %s\n\n", srcName);

    double dist;
    int *pred = NULL;
    int found = astar(G, source, target, &dist, &pred);

    printf("To %s: ", tgtName);
    if (!found || dist == INF) {
        printf("dist=inf  path=unreachable\n");
    } else {
        /* reconstruct path */
        int plen = 0;
        int *path = reconstruct_path(pred, source, target, &plen);
        if (!path) {
            printf("dist=inf  path=unreachable\n");
        } else {
            printf("dist=%g  path=", dist);
            print_path_labels(G, path, plen);
            printf("\n");
            free(path);
        }
    }

    /* cleanup */
    for (int i = 0; i < G->n; ++i) free(G->labels[i]);
    free(G->labels);
    for (int i = 0; i < G->n; ++i) free(G->adj[i].edges);
    free(G->adj);
    free(G);
    free(pred);

    return 0;
}

#endif // NODRIVER
