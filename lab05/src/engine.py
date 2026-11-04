"""Scalar reverse-mode automatic differentiation (Tasks 1-3).

Value node = data + grad + parent nodes + a local derivative rule (_backward closure).
Using the operators builds the computation graph during the forward pass; backward() walks it
in reverse topological order and ACCUMULATES (+=) into each parent's grad.
"""
import math

EPS = 1e-12


def _noop():
    pass


class Value:
    __slots__ = ("data", "grad", "_parents", "_backward", "op", "label")

    def __init__(self, data, parents=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)      # nodes this value was computed from
        self._backward = _noop              # local derivative rule, set by the op that created this node
        self.op = op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self.op}')"

    # ---------- Task 2: operations (each builds a graph node) ----------
    def __add__(self, other):
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad           # d(a+b)/da = 1
            other.grad += out.grad          # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad     # d(ab)/da = b
            other.grad += self.data * out.grad     # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "exponent must be a plain number"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad    # d(a^k)/da = k a^(k-1)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # extras needed for the moons model (sigmoid, log) ---------------------------
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, EPS)                 # clamp for numerical safety (BCE)
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += out.grad / x
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    # derived operators (built from the primitives above) ------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-_as_value(other))

    def __rsub__(self, other):
        return _as_value(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (_as_value(other) ** -1)

    def __rtruediv__(self, other):
        return _as_value(other) * (self ** -1)

    # ---------- Task 3: backward pass ----------
    def backward(self):
        order = topological_order(self)         # 1. topologically sort the graph
        self.grad = 1.0                         # 2. output gradient = 1
        for node in reversed(order):            # 3. visit in reverse order
            node._backward()                    # 4. accumulate (+=) into parents, never overwrite


def _as_value(x):
    return x if isinstance(x, Value) else Value(x, op="const")


def topological_order(root):
    """Iterative DFS post-order (parents before children). Iterative so deep graphs don't hit recursion limits."""
    order, seen = [], set()
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for p in node._parents:
            if id(p) not in seen:
                stack.append((p, False))
    return order


def vsum(values):
    """Sum a list of Values (the scalar-engine version of tensor.sum())."""
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total


def graph_summary(root):
    from collections import Counter
    nodes = topological_order(root)
    return {"nodes": len(nodes), "edges": sum(len(n._parents) for n in nodes),
            "ops": dict(Counter(n.op or "leaf" for n in nodes))}
