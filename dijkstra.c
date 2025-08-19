
#include "graph.h"

// ---------------------------------------------------------------------------
// Dijkstra
// ---------------------------------------------------------------------------

/**
 * @brief Naive Dijkstra implementation (O(n^2))
 * @param G Graph
 * @param source source node index
 * @return Path struct with all shortest paths
 */
Path run_dijkstra(Graph *G, int source) {
    int n = G->n;
    Path P;
    P.pathSize = n;
    P.path = (NodePath*)malloc(sizeof(NodePath)*n);

    int *visited = (int*)calloc(n, sizeof(int));

    for (int i = 0; i < n; i++) {
        P.path[i].d_hat = INF;
        P.path[i].Pred = -1;
    }
    P.path[source].d_hat = 0.0;

    for (int iter = 0; iter < n; iter++) {
        // Find unvisited vertex with smallest distance
        double best = INF; int u = -1;
        for (int i = 0; i < n; i++) {
            if (!visited[i] && P.path[i].d_hat < best) {
                best = P.path[i].d_hat;
                u = i;
            }
        }
        if (u == -1) break;
        visited[u] = 1;

        // Relax neighbors
        for (int j = 0; j < G->adj[u].count; j++) {
            Edge e = G->adj[u].edges[j];
            double cand = P.path[u].d_hat + e.w;
            if (cand < P.path[e.v].d_hat) {
                P.path[e.v].d_hat = cand;
                P.path[e.v].Pred = u;
            }
        }
    }

    free(visited);
    return P;
}

#ifndef NODRIVER

// ---------------------------------------------------------------------------
// Path printing
// ---------------------------------------------------------------------------

/** Recursively print path */
void print_path(const Path *P, int v, Graph *G) {
    if (P->path[v].Pred == -1) {
        printf("%s", G->labels[v]);
    } else {
        print_path(P, P->path[v].Pred, G);
        printf(" -> %s", G->labels[v]);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <file.dot> <source> [destination]\n", argv[0]);
        return 1;
    }

    Graph *G = read_dot(argv[1]);
    const char *srcName = argv[2];
    int source = -1, dest = -1;
    for (int i = 0; i < G->n; i++) {
        if (strcmp(G->labels[i], srcName) == 0) source = i;
        if (argc >= 4 && strcmp(G->labels[i], argv[3]) == 0) dest = i;
    }
    if (source < 0) { fprintf(stderr, "Source not found\n"); return 1; }
    if (argc >= 4 && dest < 0) { fprintf(stderr, "Destination not found\n"); return 1; }

    Path P = run_dijkstra(G, source);

    printf("Source: %s\n\n", srcName);
    if (argc >= 4) {
        printf("To %s: ", argv[3]);
        if (P.path[dest].d_hat == INF) {
            printf("dist=inf  path=unreachable\n");
        } else {
            printf("dist=%g  path=", P.path[dest].d_hat);
            print_path(&P, dest, G);
            printf("\n");
        }
    } else {
        for (int i = 0; i < G->n; i++) {
            printf("To %s: ", G->labels[i]);
            if (P.path[i].d_hat == INF) {
                printf("dist=inf  path=unreachable\n");
            } else {
                printf("dist=%g  path=", P.path[i].d_hat);
                print_path(&P, i, G);
                printf("\n");
            }
        }
    }

    // cleanup
    for (int i = 0; i < G->n; i++) free(G->labels[i]);
    free(G->labels);
    for (int i = 0; i < G->n; i++) free(G->adj[i].edges);
    free(G->adj);
    free(G);
    free(P.path);

    return 0;
}

#endif // NODRIVER
