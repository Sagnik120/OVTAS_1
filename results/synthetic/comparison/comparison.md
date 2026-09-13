# Benchmark Comparison: Baseline OVTAS vs. OVTAS+

**Dataset**: `synthetic`  
**Total Videos Evaluated**: 3  

## Overall Dataset Summary

| Metric | Baseline OVTAS | OVTAS+ (FAES+) | Delta |
| :--- | :---: | :---: | :---: |
| **Acc** | 14.44 | 15.56 | **+1.11** |
| **Edit** | 8.27 | 8.23 | **-0.04** |
| **F1@10** | 1.59 | 3.07 | **+1.48** |
| **F1@25** | 1.59 | 1.59 | **+0.00** |
| **F1@50** | 1.59 | 1.59 | **+0.00** |
| **Avg** | 5.49 | 6.01 | **+0.51** |

## Per-Video Breakdown

| Video ID | Acc (Base / Plus / Δ) | Edit (Base / Plus / Δ) | F1@10 (Base / Plus / Δ) | Avg (Base / Plus / Δ) |
| :--- | :---: | :---: | :---: | :---: |
| `synthetic_000` | 3.3 / 3.3 / +0.0 | 6.5 / 6.5 / +0.0 | 0.0 / 0.0 / +0.0 | 2.0 / 2.0 / +0.0 |
| `synthetic_001` | 15.0 / 16.7 / +1.7 | 13.5 / 13.5 / +0.0 | 4.8 / 4.8 / +0.0 | 8.6 / 8.9 / +0.3 |
| `synthetic_002` | 25.0 / 26.7 / +1.7 | 4.8 / 4.7 / -0.1 | 0.0 / 4.4 / +4.4 | 6.0 / 7.2 / +1.2 |
