#!/usr/bin/env python3
"""
generate_big_dot.py
-------------------
Generate a large random directed weighted graph in DOT format.

Usage:
    python generate_big_dot.py output.dot [num_nodes] [avg_out_degree]

Defaults:
    num_nodes       = 12000
    avg_out_degree  = 3

The graph is connected but has random edge directions and weights.
"""

import sys
import random

def generate_dot(num_nodes=12000, avg_out_degree=3, filename="biggraph.dot"):
    with open(filename, "w") as f:
        f.write("digraph G {\n")

        # Node labels are A0, A1, A2, ...
        nodes = [f"N{i}" for i in range(num_nodes)]

        # Ensure weak connectivity by linking i->i+1
        for i in range(num_nodes - 1):
            w = random.randint(1, 20)
            f.write(f"  {nodes[i]} -> {nodes[i+1]} [label={w}];\n")

        # Add extra random edges
        num_edges = num_nodes * avg_out_degree
        for _ in range(num_edges):
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u == v:
                continue
            w = random.randint(1, 50)
            f.write(f"  {nodes[u]} -> {nodes[v]} [label={w}];\n")

        f.write("}\n")
    print(f"DOT file with {num_nodes} nodes written to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_big_dot.py output.dot [num_nodes] [avg_out_degree]")
        sys.exit(1)

    filename = sys.argv[1]
    num_nodes = int(sys.argv[2]) if len(sys.argv) >= 3 else 12000
    avg_out_degree = int(sys.argv[3]) if len(sys.argv) >= 4 else 3

    generate_dot(num_nodes, avg_out_degree, filename)
