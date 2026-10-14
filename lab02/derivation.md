# Hand derivation (Tasks 2 and 5) — use as a checklist, rewrite in your own words/paper

Notation: batch B, X (B,64). Z1 = X W1 + b1, A1 = ReLU(Z1), Z2 = A1 W2 + b2, A2 = ReLU(Z2),
Z3 = A2 W3 + b3, P = softmax(Z3), L = -(1/B) sum_i sum_k Y_ik log P_ik.

Parameter shapes (task 2): W1 (64,W) b1 (W,) | W2 (W,W) b2 (W,) | W3 (W,10) b3 (10,)
Parameter count: W=16 -> 64*16+16 + 16*16+16 + 16*10+10 = 1,482 ; W=64 -> 4,160 + 4,160 + 650 = 8,970

Backward (chain rule, from the output back):
1. dZ3 = (P - Y)/B            <- softmax + cross-entropy collapse to this (derive it: dL/dZ3_k = P_k - Y_k)
2. dW3 = A2^T dZ3 ;  db3 = sum_rows(dZ3) ;  dA2 = dZ3 W3^T
3. dZ2 = dA2 * 1[Z2 > 0]      <- ReLU derivative is a 0/1 mask
4. dW2 = A1^T dZ2 ;  db2 = sum_rows(dZ2) ;  dA1 = dZ2 W2^T
5. dZ1 = dA1 * 1[Z1 > 0]
6. dW1 = X^T dZ1 ;  db1 = sum_rows(dZ1)
Update: theta <- theta - lr * dL/dtheta

Shape check for every line: each gradient must have the same shape as its parameter.
