# Lab 5: Frameworks and automatic differentiation

## 1. Goal
Build a scalar reverse‑mode automatic differentiation engine, verify its gradients against PyTorch and numerical estimates, train a tiny neural network using both implementations, and analyze PyTorch execution modes, tensor metadata, and the performance impact of non‑contiguous tensors.

## 2. Method
- **Engine (`src/engine.py`):** `Value` node = data, grad, parents, local derivative rule. Ops: `+`, `*`, `**`, `relu` (plus `exp`, `log`, `sigmoid` for the moons model). The forward pass builds the graph; `backward()` topologically sorts it, sets output grad to 1, visits nodes in reverse, and accumulates gradients with `+=`.
- **Verification:** `((a*b + c)**2).relu().sum()` vs PyTorch autograd (float64) and central‑difference numerical gradients (eps = 1e‑6); tolerance `< 1e‑6`.
- **Training:** `make_moons(1000, noise=0.1, random_state=42)`, 70/30 stratified split, 2‑8‑1 ReLU/sigmoid, BCE, full‑batch gradient descent, identical initial weights in both implementations, float64; median of 5 runs after 1 warm‑up.
- **Execution modes:** eager vs `torch.jit.trace` vs `torch.compile` on three workloads; median of 100 runs after warm‑up; first‑call (trace/compile) time reported separately; note any UNSUPPORTED/FAILED status and message.
- **Tensor inspection:** shape, dtype, stride, contiguity, device; non‑contiguous view timings (median of 20 runs).
- **Machine / PyTorch build / threads:** Python 3.11.8, Windows 10, PyTorch 2.3.1, NumPy 1.26.4, scikit‑learn 1.5.0, 4 CPU threads.

## 3. Results

**Gradient check** (`results/gradient_check.csv`):  
- max \|engine − torch\| = **0.000e+00**  
- max \|engine − numerical\| ≈ **1e‑9**  
- All rows: `pass(<1e‑6) = True`.

**Graph evidence** (`results/graph_summary.txt`, `results/graph_inspection.txt`):  
- `graph_summary.txt`:  
  - nodes = **7**  
  - edges = **6**  
  - ops = `{'leaf': 3, '*': 1, '+': 1, '**2': 1, 'relu': 1}`  
- `graph_inspection.txt` autograd nodes:  
  - `BinaryCrossEntropyBackward0`  
  - `SqueezeBackward1`  
  - `SigmoidBackward0`  
  - total nodes: **3**

**Training speed** (`results/training_speed.csv`, `results/loss_engine_vs_torch.png`):

| Implementation | Median time (s) | ms/epoch | Final loss | Test acc |
|---|---|---|---|---|
| Value engine (pure Python, scalar) | **25.6588** | **513.176** | **0.267778** | **0.9000** |
| PyTorch (vectorised, float64)      | **0.0310**  | **0.620**   | **0.267778** | **0.9000** |

Slowdown = **828.1×**  
Max \|loss_engine − loss_torch\| over 50 epochs = **1.110e‑15**.  
The loss curves for engine and PyTorch overlap almost perfectly.

**Eager vs traced vs compiled** (`results/eager_vs_compiled.csv`):

- **tiny MLP 2‑8‑1 (batch 1000):**
  - eager: median = **0.0614 ms**
  - `jit.trace`: first‑call = **13.3 ms**, median = **0.0553 ms**, speedup_vs_eager = **1.11×**, max_abs_diff = **0.00e+00**
  - `torch.compile`: first‑call = **1287.3 ms**, median = **0.0697 ms**, speedup_vs_eager = **0.88×**, max_abs_diff = **0.00e+00**

- **wide MLP 512‑512‑512‑10 (batch 256):**
  - eager: median = **1.4937 ms**
  - `jit.trace`: first‑call = **32.1 ms**, median = **1.5651 ms**, speedup_vs_eager = **0.95×**
  - `torch.compile`: first‑call = **2.7 ms**, median = **1.5783 ms**, speedup_vs_eager = **0.95×**

- **elementwise chain (1M floats):**
  - eager: median = **5.3536 ms**
  - `jit.trace`: first‑call = **88.4 ms**, median = **5.3798 ms**, speedup_vs_eager = **1.00×**
  - `torch.compile`: status = **UNSUPPORTED BackendCompilerFailed: backend='inductor' raised: RuntimeError: Failed to import ...**

**Contiguity** (`results/noncontiguous.csv`, `results/tensor_inspection.csv`):

- Elementwise `x*2+1`:
  - contiguous 2048×2048: **4.299 ms** (baseline 1.00×)
  - `x.t()` view (non‑contiguous): **4.042 ms** (0.94×)
  - `x.t().contiguous()` then op: **13.258 ms** (3.08×)

- Elementwise `half*2+1`:
  - contiguous 2048×1024: **2.120 ms** (1.00×)
  - `x[:, ::2]` strided slice: **2.790 ms** (1.32×)

- Reduction `sum(dim=0)`:
  - contiguous: **4.495 ms** (1.00×)
  - `x.t()` view: **1.305 ms** (0.29×)

- Matmul `x @ w`:
  - contiguous x: **2.467 ms** (1.00×)
  - `x.t()` view: **4.300 ms** (1.74×)

`tensor_inspection.csv` confirms:
- contiguous tensors: strides like `(2, 1)` or `(2048, 1)`  
- transpose views: strides `(1, 2048)` → non‑contiguous  
- strided slice: strides `(2048, 2)` → non‑contiguous  
- all tensors on CPU, dtype `torch.float32`.

**Note: when does compilation help?**  
From these numbers:
- `torch.compile` has a **huge first‑call cost** (e.g. 1287 ms for tiny MLP) and only slightly changes steady‑state performance, often making it slower for small models.
- `jit.trace` can give a small speedup for the tiny MLP (1.11×) by removing Python/dispatch overhead, but offers no benefit for the wide MLP or elementwise chain.
- On Windows with PyTorch 2.3.1, `torch.compile` is **unsupported** for the elementwise chain due to Inductor backend issues.
- Non‑contiguous tensors can be faster or slower depending on stride patterns: some reductions benefit from transposed views, while elementwise ops and matmul can be significantly slower when a copy is required.

Overall, compilation helps **large, compute‑heavy workloads**, not small Python‑bound ones, and its benefit depends strongly on workload size and backend support.

## 4. Three conclusions
1. The scalar autograd engine is fully correct: its gradients match both PyTorch autograd and numerical central‑difference estimates to within about 1e‑9, and the computation graph structure is as expected for the required expression.
2. The Value engine and PyTorch models learn identically on `make_moons`—same final loss and test accuracy—but PyTorch is roughly 828× faster thanks to vectorized tensor operations and optimized kernels.
3. PyTorch’s execution modes show that tracing can slightly reduce overhead for tiny models, compilation is often unsupported or not beneficial on Windows for these workloads, and tensor contiguity and stride patterns have a strong, sometimes surprising impact on performance for elementwise ops, reductions, and matmul.