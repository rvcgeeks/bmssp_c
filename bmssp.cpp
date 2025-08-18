/**
 * @file bmssp.cpp
 * @brief Reference-style, didactic C++11 implementation of a BMSSP-like framework with a DOT reader.
 *
 * @details
 * This single-file program demonstrates a **bounded multi-source shortest path (BMSSP)** recursion,
 * inspired by the algorithmic structure in *“Breaking the Sorting Barrier for Directed Single-Source
 * Shortest Paths”* (arXiv:2504.17033v2). It includes:
 *
 *  - A minimal directed weighted graph with a DOT reader (A -> B [label=3];).
 *  - A partial-sorting data structure `DSPartial` matching the intent of **Lemma 3.3**.
 *  - `FindPivots` (Alg. 1), `BaseCase` (Alg. 2), and `BMSSP` (Alg. 3) scaffolding with the same
 *    control-flow *structure* as the paper, adapted for clarity and small graphs.
 *  - A driver `main()` that reads a DOT file, runs BMSSP from a source, and prints distances/paths.
 *
 * ⚠️ Scope & fidelity:
 *  - This is an **educational reference** that mirrors the *structure* and key ideas (frontier S,
 *    bounded B, recursion depth l, pull-based partial ordering) rather than a tuned, asymptotically
 *    optimal research implementation. It is intentionally small, readable, and testable.
 *  - Graphs are assumed to have **non-negative weights**; behavior with negative weights is unspecified.
 *
 * ## Paper references (quotes and mapping)
 *
 * - **BMSSP subproblem & lemma (Section 3.1, Lemma 3.1)**:
 *   > “We call this subproblem bounded multi-source shortest path (BMSSP)” :contentReference[oaicite:0]{index=0}
 *
 * - **Algorithm 1 (FindPivots)**:
 *   > “Relax for k steps… If |W| > k|S| then P ← S” (shortened summary) :contentReference[oaicite:1]{index=1}
 *
 * - **Algorithm 2 (Base Case of BMSSP)**:
 *   > “Run a mini Dijkstra’s algorithm… until we find k+1 such vertices” :contentReference[oaicite:2]{index=2}
 *
 * - **Algorithm 3 (BMSSP)**:
 *   > “Pull from D a subset S_i … Recursively call BMSSP(l−1, B_i, S_i)” :contentReference[oaicite:3]{index=3}
 *
 * - **Lemma 3.3 (Partial sorting DS)**:
 *   > “Insert… Batch Prepend… Pull… in amortized O(|S'|) time” (shortened) :contentReference[oaicite:4]{index=4}
 *
 * These quotes are short (< 25 words) and the exact algorithms/lemmas above are cited to the uploaded PDF.
 *
 * Build:
 * @code
 * g++ -std=c++11 -O2 bmssp.cpp -o bmssp
 * ./bmssp graph.dot A
 * @endcode
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cmath>

/// Infinity for distances (use long long for demonstration).
static const long long INF = std::numeric_limits<long long>::max() / 4;

/** @brief Edge structure for a directed weighted graph. */
struct Edge {
    int to;
    long long w;
};

/**
 * @brief Simple directed graph with string vertex labels and integer IDs.
 * @details
 * Stores adjacency, label<->id maps, and predecessors for path reconstruction.
 */
struct Graph {
    std::vector<std::vector<Edge>> adj;
    std::unordered_map<std::string,int> id_of;
    std::vector<std::string> label_of;

    /// Predecessor to reconstruct paths (Dijkstra-like).
    std::vector<int> pred;

    int add_node(const std::string& s) {
        auto it = id_of.find(s);
        if (it != id_of.end()) return it->second;
        int id = (int)label_of.size();
        id_of[s] = id;
        label_of.push_back(s);
        adj.push_back({});
        return id;
    }
    void add_edge(int u, int v, long long w) {
        if (u >= (int)adj.size() || v >= (int)adj.size()) return;
        adj[u].push_back({v,w});
    }
    int n() const { return (int)label_of.size(); }
};

/**
 * @brief Parse a minimal DOT file format: lines like `A -> B [label=3];` or `A -> B;`
 * @details
 * - Directed edges: `A -> B [label=5];`
 * - If label absent, weight = 1
 * - Ignores subgraphs/attrs/comments; keeps it tiny and robust enough for demos.
 */
Graph parseDOT(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) {
        std::cerr << "Failed to open DOT file: " << path << "\n";
        std::exit(1);
    }
    Graph G;
    std::string line;
    while (std::getline(in, line)) {
        // Remove comments
        auto posc = line.find("//");
        if (posc != std::string::npos) line = line.substr(0, posc);

        // Trim
        auto ltrim = [](std::string &s){ s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch){return !std::isspace(ch);})); };
        auto rtrim = [](std::string &s){ s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch){return !std::isspace(ch);}).base(), s.end()); };
        ltrim(line); rtrim(line);
        if (line.empty()) continue;

        // Expect something like: A -> B [label=3];
        // Extract src, arrow, dst, optional [label=...]
        // Tokenize crudely:
        std::string src, arrow, dst, rest;
        {
            std::istringstream iss(line);
            iss >> src >> arrow >> dst;
            std::getline(iss, rest);
        }
        if (arrow != "->" || src.empty() || dst.empty()) continue;

        // strip trailing semicolon from dst or rest
        auto strip_semicolon = [](std::string& s){
            if (!s.empty() && s.back() == ';') s.pop_back();
        };
        strip_semicolon(dst);
        rtrim(dst);

        // Node labels can be quoted; remove quotes if present.
        auto unquote = [](std::string s){
            if (s.size() >= 2 && ((s.front()=='"' && s.back()=='"') || (s.front()=='\'' && s.back()=='\''))) {
                return s.substr(1, s.size()-2);
            }
            return s;
        };
        src = unquote(src);
        dst = unquote(dst);

        long long w = 1;
        // parse label if present: [label=NUMBER]
        auto lpos = rest.find("label");
        if (lpos != std::string::npos) {
            auto eq = rest.find('=', lpos);
            if (eq != std::string::npos) {
                std::string val;
                size_t i = eq+1;
                while (i < rest.size() && std::isspace(rest[i])) ++i;
                while (i < rest.size() && (std::isdigit(rest[i]) || rest[i]=='-' )) { val.push_back(rest[i]); ++i; }
                if (!val.empty()) {
                    w = std::atoll(val.c_str());
                    if (w < 0) {
                        std::cerr << "Warning: negative weight read; algorithm assumes non-negative.\n";
                    }
                }
            }
        }

        int u = G.add_node(src);
        int v = G.add_node(dst);
        G.add_edge(u,v,w);
    }
    return G;
}

/** @brief Lightweight struct holding algorithm state for the demo. */
struct AlgoState {
    Graph* G;
    std::vector<long long> dist; ///< \f$\hat{d}[\cdot]\f$ current distance estimates.
    std::vector<bool> complete;  ///< whether a vertex is "complete" (distance fixed).
    int k;                       ///< parameter k \f$\approx \lfloor \log^{1/3} n \rfloor\f$
    int t;                       ///< parameter t \f$\approx \lfloor \log^{2/3} n \rfloor\f$
};

/**
 * @brief Partial-sorting data structure D (Lemma 3.3; block-queue flavor).
 *
 * @details
 * **Lemma 3.3 (quoted)**:
 * > “Insert… Batch Prepend… Pull… in amortized O(|S'|) time.” :contentReference[oaicite:5]{index=5}
 *
 * We implement:
 *  - `Initialize(M,B)`: set block cap M and global upper bound B.
 *  - `Insert(key,value)`: insert/update; goes to D1 (regular inserts).
 *  - `BatchPrepend(L)`: prepend smaller-than-current-min into D0 (batched).
 *  - `Pull()`: return up to M smallest keys and a separating bound x.
 *
 * This is a compact educational version consistent with the behavior used by Alg. 3.
 */
class DSPartial {
public:
    struct Pair { int key; long long val; };
    DSPartial(): M(1), B_ub(INF) {}

    void Initialize(size_t m, long long B) {
        M = std::max<size_t>(1, m);
        B_ub = B;
        D0.clear();
        D1.clear();
        min_current = B_ub;
        pos.clear();
    }

    void Insert(int key, long long val) {
        // Keep best value per key.
        auto it = pos.find(key);
        if (it != pos.end()) {
            long long old = it->second;
            if (val >= old) return; // keep smaller
            it->second = val;
        } else {
            pos[key] = val;
        }
        D1.push_back({key, val});
        if (val < min_current) min_current = val;
    }

    void BatchPrepend(const std::vector<Pair>& L) {
        if (L.empty()) return;
        for (auto &p : L) {
            auto it = pos.find(p.key);
            if (it != pos.end()) {
                if (p.val < it->second) {
                    it->second = p.val;
                    D0.push_back(p);
                    if (p.val < min_current) min_current = p.val;
                }
            } else {
                pos[p.key] = p.val;
                D0.push_back(p);
                if (p.val < min_current) min_current = p.val;
            }
        }
    }

    /** @brief Pull up to M smallest keys and return (S, x) with x as separating bound. */
    std::pair<std::vector<int>, long long> Pull() {
        // Collect candidates from both D0 and D1, but only the keys with their best values.
        if (pos.empty()) return std::make_pair(std::vector<int>{}, B_ub);

        // Materialize (key,val) of current map, take up to M with smallest val.
        std::vector<Pair> all;
        all.reserve(pos.size());
        for (auto &kv : pos) all.push_back({kv.first, kv.second});
        std::nth_element(all.begin(), all.begin() + std::min((size_t)M, all.size()) , all.end(),
                         [](const Pair& a, const Pair& b){ return a.val < b.val; });
        // We still need them sorted for deterministic S selection and x computation.
        std::sort(all.begin(), all.end(), [](const Pair& a, const Pair& b){ return a.val < b.val; });

        size_t take = std::min((size_t)M, all.size());
        std::vector<int> S; S.reserve(take);
        for (size_t i=0;i<take;i++) {
            S.push_back(all[i].key);
            pos.erase(all[i].key); // removed from structure
        }

        long long x;
        if (take == all.size()) {
            x = B_ub; // nothing remains
            min_current = B_ub;
        } else {
            x = all[take].val; // separating bound: max(S) < x <= min(remaining)
            // update min_current to next minimal
            long long next_min = B_ub;
            for (size_t i=take;i<all.size();++i) {
                if (all[i].val < next_min) next_min = all[i].val;
            }
            min_current = next_min;
        }
        // lazily keep D0/D1 vectors; pos is authoritative.
        return std::make_pair(S, x);
    }

    bool empty() const { return pos.empty(); }

private:
    size_t M;
    long long B_ub;
    long long min_current;

    // We model D0 / D1 just to reflect two insertion modes; pos stores current best per key.
    std::vector<Pair> D0, D1;
    std::unordered_map<int,long long> pos;
};

/** @brief Helper: push a neighbor relaxation with the “≤” tie rule (Remark 3.4). */
static inline bool relax_leq(long long du, long long wuv, long long &dv, int u, int v, std::vector<int> &pred) {
    // “conditions are d̂[u] + w_uv ≤ d̂[v]… so an edge relaxed on a lower level can be re-used”
    // (short quote per Remark 3.4) :contentReference[oaicite:6]{index=6}
    if (du == INF) return false;
    long long cand = du + wuv;
    if (cand <= dv) { dv = cand; pred[v] = u; return true; }
    return false;
}

/**
 * @brief Algorithm 2: BaseCase(B,S) — Dijkstra from a singleton until k+1 found or exhausted.
 * @details
 * **Quote (Alg. 2)**:
 * > “Run a mini Dijkstra’s algorithm… until we find k+1 such vertices” :contentReference[oaicite:7]{index=7}
 *
 * This base case expects @p S to be a singleton (we enforce it in the caller).
 * It collects up to k+1 vertices with distances < B reachable via the start x∈S.
 *
 * @param st  Shared algorithm state (dist, complete, params).
 * @param B   Upper bound \f$B\f$ (stop if tentative distances reach \f$\ge B\f$).
 * @param S   Set of frontier vertices (singleton here).
 * @return pair(Bprime, U) as in Alg. 2.
 */
static std::pair<long long, std::vector<int>>
BaseCase(AlgoState &st, long long B, const std::vector<int> &S) {
    assert(S.size() == 1 && "BaseCase expects singleton S");
    int x = S[0];

    // A tiny Dijkstra restricted by the bound B.
    struct Node { long long d; int v; };
    auto cmp = [](const Node& a, const Node& b){ return a.d > b.d; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);

    std::vector<bool> in_heap(st.G->n(), false);
    pq.push({st.dist[x], x});
    in_heap[x] = true;

    std::vector<int> U0;
    // Push until we collect k+1 or heap empty.
    while (!pq.empty() && (int)U0.size() < st.k + 1) {
        auto cur = pq.top(); pq.pop();
        in_heap[cur.v] = false;
        if (st.complete[cur.v]) continue; // if already fixed by outer calls
        // Only accept < B
        if (cur.d >= B) break;
        // mark discovered for U0 and relax outgoing
        U0.push_back(cur.v);
        for (const auto &e : st.G->adj[cur.v]) {
            long long old = st.dist[e.to];
            if (relax_leq(st.dist[cur.v], e.w, st.dist[e.to], cur.v, e.to, st.G->pred)) {
                if (!in_heap[e.to]) { pq.push({st.dist[e.to], e.to}); in_heap[e.to] = true; }
                else { /* conceptually decrease-key via push; fine for demo */ pq.push({st.dist[e.to], e.to}); }
            } else {
                // If cand < B but not improving dv, still allow heap presence; the ≤ rule handles ties.
                (void)old;
            }
        }
    }

    if ((int)U0.size() <= st.k) {
        // return B' = B, U = U0
        return std::make_pair(B, U0);
    } else {
        // return B' = max d[v] over U0, U = those with d[v] < B'
        long long Bp = 0;
        for (int v : U0) Bp = std::max(Bp, st.dist[v]);
        std::vector<int> U;
        for (int v : U0) if (st.dist[v] < Bp) U.push_back(v);
        return std::make_pair(Bp, U);
    }
}

/**
 * @brief Algorithm 1: FindPivots(B,S) — relax k steps to build W; if |W|>k|S| set P=S.
 *
 * @details
 * **Quote (Alg. 1)**:
 * > “Relax for k steps… If |W| > k|S| then P ← S” (shortened) :contentReference[oaicite:8]{index=8}
 *
 * For this didactic version:
 *  - We simulate k relaxation rounds from currently-known distances, gated by B.
 *  - W collects vertices reached in these rounds with tentative distance < B.
 *  - If W grew “too large”, P = S; else pick P as the subset of S with largest local spread
 *    (heuristic consistent with |P| ≤ |W|/k bound).
 *
 * @param st  Shared algorithm state.
 * @param B   Upper bound \f$B\f$.
 * @param S   Current frontier set.
 * @return pair(P, W).
 */
static std::pair<std::vector<int>, std::vector<int>>
FindPivots(AlgoState &st, long long B, const std::vector<int> &S) {
    std::unordered_set<int> Wset(S.begin(), S.end());
    std::vector<int> frontier = S;

    for (int step = 1; step <= st.k; ++step) {
        std::vector<int> next;
        for (int u : frontier) {
            for (const auto &e : st.G->adj[u]) {
                if (st.dist[u] == INF) continue;
                long long cand = st.dist[u] + e.w;
                if (cand < B) {
                    // Even if not strictly improving, consider for W exploration as in Alg.1 lines 7-10.
                    if (relax_leq(st.dist[u], e.w, st.dist[e.to], u, e.to, st.G->pred)) {
                        next.push_back(e.to);
                        Wset.insert(e.to);
                    } else {
                        // if not improved but within bound, still include for coverage heuristics
                        next.push_back(e.to);
                        Wset.insert(e.to);
                    }
                }
            }
        }
        frontier.swap(next);
        if ((int)Wset.size() > st.k * (int)S.size()) {
            // Early return: P = S
            std::vector<int> P = S;
            std::vector<int> W(Wset.begin(), Wset.end());
            return std::make_pair(P, W);
        }
    }

    // Otherwise derive a small P ⊆ S. Heuristic: pick ≤ |W|/k highest-degree (proxy) or farthest in dist.
    // For demo: choose up to max(1, |W|/k) elements with smallest current dist (to emulate “pivots”).
    std::vector<int> W(Wset.begin(), Wset.end());
    size_t Plim = std::max<size_t>(1, W.size() / std::max(1, st.k));
    std::vector<int> P = S;
    std::sort(P.begin(), P.end(), [&](int a, int b){ return st.dist[a] < st.dist[b]; });
    if (P.size() > Plim) P.resize(Plim);
    return std::make_pair(P, W);
}

/**
 * @brief Algorithm 3: BMSSP(l, B, S) — recursive bounded multi-source shortest path.
 *
 * @details
 * **Quote (Alg. 3, Section 4 of arXiv:2504.17033v2)**:
 * > “Pull from D a subset S_i … Recursively call BMSSP(l−1, B_i, S_i).”
 *
 * High-level reproduction of Algorithm 3:
 *  1. If l = 0, return BaseCase(B,S) (Algorithm 2).
 *  2. (P,W) ← FindPivots(B,S).
 *  3. Initialize partial-sorting structure D with M = 2^{l−1}·t.
 *  4. Insert ⟨x,d̂[x]⟩ for x∈P into D.
 *  5. While |U| < k·2^l·t and D non-empty:
 *       - Pull (S_i,B_i);
 *       - (B’_i,U_i) ← BMSSP(l−1,B_i,S_i);
 *       - Relax edges from U_i into D (Insert/BatchPrepend).
 *  6. Return B’ ← min(B’_i,B), and U augmented with W.
 *
 * ---
 *
 * ### About hyperparameters k and t
 * - **In the original demo implementation (tiny graphs)**:
 *   \code
 *   st.k = std::max(1, (int)std::cbrt(n));          // k ~ n^{1/3}
 *   st.t = std::max(1, (int)std::sqrt(std::sqrt(n))); // t ~ n^{1/4}
 *   \endcode
 *   This was sufficient for n ≈ 5–10, but caused premature cutoffs
 *   for larger graphs (distances left INF).
 *
 * - **In the improved adaptive version (robust for small & large graphs)**:
 *   \code
 *   st.k = std::max(2, (int)std::sqrt(n));          // breadth ~ sqrt(n)
 *   st.t = std::max(2, (int)std::ceil(log2(n)));    // depth ~ log n
 *   \endcode
 *   These scale gracefully:
 *     - Small n (10): k≈3, t≈4.
 *     - Medium n (60): k≈7, t≈6.
 *     - Large n (1000): k≈31, t≈10.
 *
 * Thus BMSSP explores enough vertices to approximate full shortest paths,
 * without hardcoding tiny constants.
 *
 * @param st Shared algorithmic state (graph, dist, pred, complete[]).
 * @param l  Recursion level (0 = BaseCase).
 * @param B  Upper distance bound.
 * @param S  Frontier subset.
 * @return Pair(B’, U) where B’ is updated bound and U is the set of
 *         completed vertices at this recursion.
 */
static std::pair<long long, std::vector<int>>
BMSSP(AlgoState &st, int l, long long B, const std::vector<int> &S) {
    if (S.empty()) return std::make_pair(B, std::vector<int>{});
    // Level 0: BaseCase
    if (l == 0) {
        // Algorithm 2 (BaseCase) is defined only for singletons S = {s}.
        // If multiple vertices are present, run BaseCase once per singleton
        // and merge results. This avoids assertion failure.
        long long Bprime = B;
        std::vector<int> U;
        for (int s : S) {
            std::vector<int> singleton = {s};
            auto sub = BaseCase(st, B, singleton);
            Bprime = std::min(Bprime, sub.first);
            U.insert(U.end(), sub.second.begin(), sub.second.end());
        }
        return std::make_pair(Bprime, U);
    }

    // 1) Find pivots
    std::vector<int> P, W;
    {
        auto pw = FindPivots(st, B, S);
        P = pw.first; W = pw.second;
    }

    // 2) Init D with M = 2^{l-1} * t  (small integer cap for demo)
    size_t M = (size_t)std::max(1, (1 << std::max(0, l-1))) * (size_t)std::max(1, st.t);
    DSPartial D;
    D.Initialize(M, B);
    for (int x : P) D.Insert(x, st.dist[x]);

    long long B0 = (P.empty() ? B : st.dist[P[0]]);
    for (int x : P) B0 = std::min(B0, st.dist[x]);

    std::vector<int> U; U.reserve(S.size() + 8);
    long long Bprime = B;

    // limit: |U| < k * 2^l * t   (mirrors |U| < k·2^{l} t in Alg. 3)
    size_t U_limit = (size_t)std::max(1, st.k) * (size_t)std::max(1, (1 << l)) * (size_t)std::max(1, st.t);

    while (U.size() < U_limit && !D.empty()) {
        // 3) Pull a small subset and a separating bound
        std::vector<int> Si;
        long long Bi;
        {
            auto out = D.Pull();
            Si = out.first;
            Bi = out.second;
        }
        if (Si.empty()) break;

        // 4) Recurse
        auto sub = BMSSP(st, l-1, Bi, Si);
        long long Bpi = sub.first;
        const std::vector<int> &Ui = sub.second;
        // Add Ui to U; mark complete
        for (int u : Ui) {
            if (!st.complete[u]) st.complete[u] = true;
            U.push_back(u);
        }

        // 5) Relax from Ui and feed D based on interval membership
        std::vector<DSPartial::Pair> K; K.reserve(Ui.size());
        for (int u : Ui) {
            for (const auto &e : st.G->adj[u]) {
                long long olddv = st.G->label_of.size() ? st.G->label_of.size() : 0; (void)olddv;
                long long before = st.dist[e.to];
                bool improved = relax_leq(st.dist[u], e.w, st.dist[e.to], u, e.to, st.G->pred);
                long long val = st.dist[e.to];
                // We need to decide {Insert, BatchPrepend} w.r.t. [B'_i, B_i) vs [B_i,B)
                if (val >= Bi && val < B) {
                    D.Insert(e.to, val);
                } else if (val >= Bpi && val < Bi) {
                    K.push_back({e.to, val});
                }
            }
        }
        // BatchPrepend K ∪ {x∈Si : d̂[x]∈[B'_i, B_i)}
        for (int x : Si) {
            if (st.dist[x] >= Bpi && st.dist[x] < Bi) K.push_back({x, st.dist[x]});
        }
        D.BatchPrepend(K);

        // Update Bprime to reflect progress (min of returned B'_i and B)
        Bprime = std::min(Bprime, Bpi);
    }

    // Add W vertices that are already complete with d < B'
    for (int x : W) {
        if (!st.complete[x] && st.dist[x] < Bprime) {
            st.complete[x] = true;
            U.push_back(x);
        }
    }
    return std::make_pair(Bprime, U);
}

/**
 * @brief Utility: run full SSSP by driving BMSSP from (l = ceil(log n / t), S={s}, B=INF).
 *
 * @details
 * The top-level invocation mirrors the statement:
 * > “main algorithm calls BMSSP with parameters l = ⌈(log n)/t⌉, S = {s}, B = ∞” :contentReference[oaicite:10]{index=10}
 *
 * On small graphs, we approximate \f$l = \max(1,\lceil \log_2(n)/t \rceil)\f$.
 */
static void run_bmssp_all(Graph &G, int s_id) {
    AlgoState st;
    st.G = &G;
    int n = G.n();
    st.k = std::max(1, (int)std::cbrt(std::max(1, n))); // crude k ~ n^{1/3} for tiny demos
    st.t = std::max(1, (int)std::sqrt(std::max(1, (int)std::sqrt(n)))); // tiny t for demos

    G.pred.assign(n, -1);
    st.dist.assign(n, INF);
    st.complete.assign(n, false);
    st.dist[s_id] = 0;
    st.complete[s_id] = false; // will be set as discovered in BMSSP
    G.pred[s_id] = s_id;

    int l = std::max(1, (int)((std::log(std::max(2, n)) / std::log(2.0)) / std::max(1, st.t)));

    std::vector<int> S = {s_id};
    long long B = INF;
    auto res = BMSSP(st, l, B, S);
    (void)res; // B' not used directly here

    // Mark source complete if reachable
    st.complete[s_id] = (st.dist[s_id] == 0);

    // Output distances and predecessor paths
    std::cout << "Source: " << G.label_of[s_id] << "\n\n";
    for (int v = 0; v < n; ++v) {
        std::cout << "To " << G.label_of[v] << ": ";
        if (st.dist[v] >= INF/2) {
            std::cout << "dist=INF  path=" << G.label_of[v] << "\n";
            continue;
        }
        std::cout << "dist=" << st.dist[v] << "  path=";
        // reconstruct
        std::vector<int> path;
        int cur = v, guard = 0;
        while (cur != -1 && guard++ < n+5) {
            path.push_back(cur);
            if (cur == G.pred[cur]) break;
            cur = G.pred[cur];
        }
        if (path.empty() || path.back() != s_id) {
            // fallback: try to chase until source if possible
            std::reverse(path.begin(), path.end());
        } else {
            std::reverse(path.begin(), path.end());
        }
        for (size_t i=0;i<path.size();++i) {
            if (i) std::cout << " -> ";
            std::cout << G.label_of[path[i]];
        }
        std::cout << "\n";
    }
}

/** @brief Pretty-print a tiny usage message. */
static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <graph.dot> <source_label>\n";
    std::cerr << "DOT edges like:  A -> B [label=3];\n";
}

/**
 * @brief Example driver with two modes:
 * @details
 * - `bmssp_dot.exe graph.dot SRC` : runs BMSSP from SRC and prints all shortest paths/distances.
 * - `bmssp_dot.exe graph.dot SRC DST` : runs BMSSP from SRC and prints only the path SRC→DST.
 */
int main(int argc, char** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }
    std::string dot = argv[1];
    std::string src = argv[2];
    std::string dst;
    bool singleTarget = false;
    if (argc >= 4) { dst = argv[3]; singleTarget = true; }

    Graph G = parseDOT(dot);
    auto it = G.id_of.find(src);
    if (it == G.id_of.end()) {
        std::cerr << "Source label '" << src << "' not found in DOT.\n";
        return 1;
    }
    int s_id = it->second;

    // Run BMSSP from source
    AlgoState st;
    st.G = &G;
    int n = G.n();
    
    st.t = std::max(2, (int)std::ceil(std::log2(std::max(2, n)))); // depth ~ log n
    st.k = std::max(2, (int)std::sqrt(n));                         // breadth ~ sqrt n
    // st.k = std::max(1, (int)std::cbrt(std::max(1, n)));
    // st.t = std::max(1, (int)std::sqrt(std::max(1, (int)std::sqrt(n))));
    
    G.pred.assign(n, -1);
    st.dist.assign(n, INF);
    st.complete.assign(n, false);
    st.dist[s_id] = 0;
    G.pred[s_id] = s_id;
    int l = std::max(1, (int)((std::log(std::max(2, n)) / std::log(2.0)) / std::max(1, st.t)));
    std::vector<int> S = {s_id};
    (void)BMSSP(st, l, INF, S);

    if (!singleTarget) {
        // Print all
        std::cout << "Source: " << G.label_of[s_id] << "\n\n";
        for (int v = 0; v < n; ++v) {
            std::cout << "To " << G.label_of[v] << ": ";
            if (st.dist[v] >= INF/2) {
                std::cout << "dist=INF  path=" << G.label_of[v] << "\n";
                continue;
            }
            std::cout << "dist=" << st.dist[v] << "  path=";
            // reconstruct path
            std::vector<int> path;
            int cur = v, guard = 0;
            while (cur != -1 && guard++ < n+5) {
                path.push_back(cur);
                if (cur == G.pred[cur]) break;
                cur = G.pred[cur];
            }
            std::reverse(path.begin(), path.end());
            for (size_t i=0;i<path.size();++i) {
                if (i) std::cout << " -> ";
                std::cout << G.label_of[path[i]];
            }
            std::cout << "\n";
        }
    } else {
        // Print only source→dst
        auto jt = G.id_of.find(dst);
        if (jt == G.id_of.end()) {
            std::cerr << "Target label '" << dst << "' not found in DOT.\n";
            return 1;
        }
        int d_id = jt->second;
        std::cout << "Source: " << src << ", Target: " << dst << "\n";
        if (st.dist[d_id] >= INF/2) {
            std::cout << "No path found.\n";
        } else {
            std::cout << "dist=" << st.dist[d_id] << "  path=";
            std::vector<int> path;
            int cur = d_id, guard = 0;
            while (cur != -1 && guard++ < n+5) {
                path.push_back(cur);
                if (cur == G.pred[cur]) break;
                cur = G.pred[cur];
            }
            std::reverse(path.begin(), path.end());
            for (size_t i=0;i<path.size();++i) {
                if (i) std::cout << " -> ";
                std::cout << G.label_of[path[i]];
            }
            std::cout << "\n";
        }
    }
    return 0;
}
