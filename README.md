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

- **Other References**:  
  - [LinkedIn Post](https://www.linkedin.com/posts/diogo-ribeiro-9094604a_graphalgorithms-computerscience-datastructures-activity-7361523109146910720-f0Ix)  
  - [Medium Article](https://medium.com/@teggourabdenour/deconstructing-the-shortest-path-algorithm-a-deep-dive-into-theory-vs-implementation-3c6c8149ac16)  
  - [GitHub Reference Repo](https://github.com/madaffrager/Bounded-Multi-Source-Shortest-Path-Algorithm)  
  - [Rust Crate bmssp](https://lib.rs/crates/bmssp)

---

## 🛠 Compilation

You need a **C++11 or later** compiler.

```bash
gcc -O2 bmssp.c -o bmssp
````

---

## ▶️ Running

The program expects a `.dot` file containing a directed graph with edge weights encoded as `label="w"`.

```bash
./bmssp <file.dot> <source_label> [dest_label]
```

Example output:

```
> bmssp.exe sample_big.dot N0
Source: N0

To N0: dist=0  path=N0
To N1: dist=4  path=N0 -> N1
To N2: dist=11  path=N0 -> N1 -> N2
To N3: dist=14  path=N0 -> N1 -> N2 -> N3
To N4: dist=20  path=N0 -> N1 -> N2 -> N3 -> N4
To N5: dist=7  path=N0 -> N5
To N6: dist=15  path=N0 -> N5 -> N6
To N7: dist=17  path=N0 -> N5 -> N6 -> N7
To N8: dist=26  path=N0 -> N5 -> N6 -> N7 -> N8
To N9: dist=30  path=N0 -> N5 -> N6 -> N7 -> N8 -> N9
To N10: dist=17  path=N0 -> N5 -> N10
...
```

---

## ⚙️ Hyperparameters

* **t**: recursion depth, set adaptively ≈ `(log2(n))^(2/3)`
* **k**: frontier breadth, set adaptively ≈ `(log2(n))^(1/3)`

These scale automatically with graph size to balance performance and completeness.

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
