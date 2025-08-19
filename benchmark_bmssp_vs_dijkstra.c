/**
 * @file benchmark_bmssp_vs_dijkstra.c
 * @brief Benchmark BMSSP vs Dijkstra on DOT graph input.
 *
 * Usage:
 *   benchmark_bmssp_vs_dijkstra.exe graph.dot SourceNode [runs]
 *
 * - Reads graph from DOT file.
 * - Runs both BMSSP and Dijkstra from given source node.
 * - Measures runtimes in milliseconds using CLOCK_MONOTONIC.
 * - Repeats experiment N times (default 5 if not given).
 * - Reports average runtimes and speedup factor.
 *
 * No correctness verification here, only performance.
 */

#include <time.h>

#define DEFAULT_RUNS 10
#define NODRIVER

#include "bmssp.c"
#include "dijkstra.c"

/* Utility: current time in ms */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s graph.dot SourceNode [DestNode] [runs]\n", argv[0]);
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

    int runs = (argc >= 5) ? atoi(argv[4]) : DEFAULT_RUNS;
    if (runs <= 0) runs = DEFAULT_RUNS;

    double total_dijkstra = 0.0, total_bmssp = 0.0, dt1 = 0.0, dt2 = 0.0;

    for (int r = 0; r < runs; r++) {
        double t0, t1;

        t0 = now_ms();
        Path P1 = run_dijkstra(G, source);
        t1 = now_ms();
        dt1 = t1 - t0;
        total_dijkstra += dt1;
        free(P1.path);

        t0 = now_ms();
        Path P2 = run_BMSSP(G, source);
        t1 = now_ms();
        dt2 = t1 - t0;
        total_bmssp += dt2;
        free(P2.path);
        
        printf("  Run : %d\n", r + 1);
        printf("  Dijkstra : %.3f ms\n", dt1);
        printf("  BMSSP    : %.3f ms\n", dt2);
    }

    double avg_dijkstra = total_dijkstra / runs;
    double avg_bmssp    = total_bmssp / runs;
    double percent      = (avg_dijkstra > 0.0 ? 
                           ((avg_dijkstra - avg_bmssp) / avg_dijkstra) * 100.0 
                           : 0.0);

    printf("Benchmark results over %d runs (Source=%s):\n", runs, srcName);
    printf("  Dijkstra avg: %.3f ms\n", avg_dijkstra);
    printf("  BMSSP    avg: %.3f ms\n", avg_bmssp);
    printf("  Percent Fast: %.1f%%\n", percent);

    // cleanup
    for (int i = 0; i < G->n; i++) free(G->labels[i]);
    free(G->labels);
    for (int i = 0; i < G->n; i++) free(G->adj[i].edges);
    free(G->adj);
    free(G);

    return 0;
}
