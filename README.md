CUDA Stream Compaction
======================

This project implements different GPU stream compaction approaches in CUDA, from scratch.

This project is forked from **University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

## Parallel Scan

// TODO: Introduce parallel scan

// TODO: Naive approach

// TODO: Work efficient approach +  Shared Memory && Hardware Optimization

In this project, we provide multiple types of Parallel Scan implementation as below, also included thrust implementation for comparison:
- CPU based: It will be used as the expected value for the other tests.
- Naive approach
  - The algorithm performs `O(n log2 n)` addition operations.
- Work-Efficient approach
  - The algorithm performs `O(n)` operations
- Thrust: Call the `thrust` function directly, for performance comparison.

## Stream Compaction

Informally, stream compaction is a filtering operation: from an input vector, it selects a subset of this vector and packs that subset into a dense output vector. 
![](img/example.jpg)
More formally, stream compaction takes an input vector vi and a predicate p, and outputs only those elements in vi for which p(vi ) is true, preserving the ordering of the input elements.
(From GPU Gems 3, Chapter 39)

// TODO: connect parallal scan with stream compaction

## Performance Analysis

// Diagram

## Reference
- GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

