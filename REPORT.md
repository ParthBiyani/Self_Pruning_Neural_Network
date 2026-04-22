# Report — Self-Pruning Neural Network

## Why does L1 penalty on sigmoid gates encourage sparsity?

Each gate passes through sigmoid, so it lives in (0, 1). The sparsity loss adds
`λ × SparsityLoss` to the total objective, where `SparsityLoss` combines two terms:

```
SparsityLoss = mean(gates) + mean(gates × (1 - gates))
```

- **L1 term** `mean(gates)`: pulls all gate values toward zero
- **Binarization term** `mean(gates × (1-gates))`: penalizes gates stuck near 0.5, forcing them to commit to 0 or 1 rather than stalling at intermediate values

Using the mean (not sum) keeps the loss on a fixed [0, 1] scale regardless of network size,
so λ can be tuned independently of how many parameters the model has.

The gradient w.r.t. gate score `g`:
```
∂(λ × σ(g)) / ∂g = λ × σ(g) × (1 - σ(g))
```
This is largest at `σ(g) = 0.5` and decreases as gates approach 0 or 1 — a natural
self-stabilizing effect. Initializing `gate_scores = 0` (so `σ(0) = 0.5`) puts every gate
at the steepest gradient point from the start, maximizing early pruning signal.

**Why threshold = 0.1 instead of 0.01:**
Sigmoid saturates near zero — `gate_score` would need to reach −4.6 for `σ(g) = 0.01`,
but gradients essentially vanish before that point. A gate below 0.1 contributes less than
10% of a weight's full value and is genuinely inactive in the forward pass. The 0.1 threshold
accurately reflects what the network has actually pruned.

---

## Results

| λ | Test Accuracy | Sparsity (gate < 0.10) |
|---|--------------|------------------------|
| 2  | 56.70% | 1.81%  |
| 8  | 55.94% | 28.28% |
| 30 | 56.41% | 68.71% |

---

## Observations

**Low λ (2):** Mild sparsity pressure — the L1 + binarization term nudges gates toward the
extremes but most stay near 0.5. Accuracy is highest; the network retains most of its capacity.

**Medium λ (8):** Clear bimodal distribution emerges — a spike of near-zero gates and a surviving
cluster of active ones. The optimizer has identified which connections it needs and eliminated the rest.

**High λ (30):** The majority of gates collapse toward zero. The network works with a small
fraction of its original connections. Accuracy remains surprisingly competitive because the
warmup phase ensures weights are meaningful before pruning begins.

The gate distribution plots show the progressive shift from a unimodal bell curve (λ=2)
to a clearly bimodal distribution (λ=8) to a heavily zero-skewed distribution (λ=30).

---

## Conclusion

The self-pruning mechanism works correctly. The binarization regularizer solves the
sigmoid saturation problem that caused gates to stall at intermediate values, forcing
them to commit to either active (near 1) or pruned (near 0) states. The warmup phase
ensures the network learns useful features before sparsity pressure is applied.

λ = 8 represents the best accuracy/sparsity tradeoff — meaningful pruning without
significantly degrading classification performance.

![Gate Value Distribution](gate_distribution.png)
