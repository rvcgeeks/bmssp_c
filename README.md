# BMSSP_CPP

C++11 reference implementation of the **Bounded Multi-Source Shortest Path (BMSSP)** algorithm from  
[*Breaking the Sorting Barrier for Directed Single-Source Shortest Paths*](https://arxiv.org/pdf/2504.17033).

---

## 🚀 About this Project

This repository provides a **from-scratch implementation** of BMSSP in **one C++11 file**, capable of reading weighted directed graphs from `.dot` files and computing approximate shortest paths.  

⚠️ **Disclaimer**: This code was created with the help of [ChatGPT](https://chatgpt.com/share/68a38505-bc8c-8010-8a36-9586d2a481a7) and is open for:
- Corrections  
- Testing  
- Benchmarking  
- Collaboration  

It is not yet optimized for production use, but aims to be a faithful educational implementation closely referencing the original paper.

---

## 📄 References

- **Original Paper**:  
  [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/pdf/2504.17033)

- **Implementation from ChatGPT**:  
  [Shared ChatGPT conversation](https://chatgpt.com/share/68a38505-bc8c-8010-8a36-9586d2a481a7)

- **Other References**:  
  - [LinkedIn Post](https://www.linkedin.com/posts/george-pashev-04485635_tsinghua-university-reportedly-breaks-the-activity-7361081529038426114-NyzU)  
  - [Medium Article](https://medium.com/@teggourabdenour/deconstructing-the-shortest-path-algorithm-a-deep-dive-into-theory-vs-implementation-3c6c8149ac16)  
  - [GitHub Reference Repo](https://github.com/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm)  
  - [Rust Crate bmssp](https://lib.rs/crates/bmssp)

---

## 🛠 Compilation

You need a **C++11 or later** compiler.

```bash
g++ -std=c++11 -O2 bmssp_dot.cpp -o bmssp
````

---

## ▶️ Running

The program expects a `.dot` file containing a directed graph with edge weights encoded as `label="w"`.

```bash
./bmssp sample_weighted.dot
```

Example output:

```
Graph has 60 nodes
BMSSP params: t=6, k=7
Source: N0

To N0: dist=0   path=N0
To N1: dist=4   path=N0 -> N1
To N2: dist=11  path=N0 -> N1 -> N2
...
To N25: dist=42 path=N0 -> N5 -> N10 -> N15 -> N20 -> N25
To N26: dist=46 path=N0 -> N5 -> N10 -> N15 -> N20 -> N25 -> N26
...
```

---

## ⚙️ Hyperparameters

* **t**: recursion depth, set adaptively ≈ `log2(n)`
* **k**: frontier breadth, set adaptively ≈ `sqrt(n)`

These scale automatically with graph size to balance performance and completeness.

In the original demo version, much smaller values were used (`n^{1/3}`, `n^{1/4}`) which worked only for toy graphs. This repo uses adaptive scaling to handle both **small** and **large** graphs.

---

## 🤝 Contributing

Contributions are welcome!
Please open issues or PRs for:

* Correctness fixes
* Performance optimizations
* Parallelization
* Better DOT parsing or visualization tools

---

## 📜 License

This implementation is released under the MIT License.
