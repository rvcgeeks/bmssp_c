
#ifndef GRAPH_H
#define GRAPH_H

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
// Path struct returned by run_(pathfinding algorithm) function
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

#endif // GRAPH_H
