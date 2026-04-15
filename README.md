<p align="center">
  <img src="docs/logo.png" width="220" alt="Spikenaut">
</p>

<h1 align="center">spikenaut-router</h1>
<p align="center">SNN-based sparse domain routing — the Anti-Hallucination Layer</p>

<p align="center">
  <a href="https://crates.io/crates/spikenaut-router"><img src="https://img.shields.io/crates/v/spikenaut-router" alt="crates.io"></a>
  <a href="https://docs.rs/spikenaut-router"><img src="https://docs.rs/spikenaut-router/badge.svg" alt="docs.rs"></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-orange" alt="GPL-3.0">
</p>

---

A small LIF network acts as a neural router: it classifies incoming signals into domain
categories and activates only the relevant processing pipelines. Only neurons that
actually fire trigger downstream computation — sparse activation by design.

## Features

- `AhlRouter` — 3-neuron LIF bank routing over `Chemistry`, `Mathematics`, `DigitalLogic`
- `DomainSignals::from_text(text)` — keyword-density feature extraction
- `RoutingDecision` — activation mask + per-domain confidence scores
- STDP-based online refinement (correct dispatches potentiate, failures depress)
- Winner-take-all lateral inhibition between domain neurons
- Configurable `MIN_FIRE_RATE` sparse-activation floor
- **CSR Sparse Synaptic Map** — Compressed Sparse Row format for Blackwell-optimized GPU execution (20× VRAM reduction at 5% sparsity)
- **SAAQ Telemetry Integration** — adaptation-aware routing that steers traffic away from exhausted neurons toward fresh ones
- **Ballast-Lab Feedback Loop** — CSV telemetry export for Julia symbolic regression to discover optimal routing policies
- **RoutingPolicy** — configurable policy equation (`α·spikes − β·adaptation − γ·quant_error + δ·base_weight`)

## Installation

```toml
spikenaut-router = "0.1"
```

## Quick Start

```rust
use spikenaut_router::AhlRouter;

let mut router = AhlRouter::new();

let decision = router.route("solve the differential equation dy/dx = sin(x)");
// decision.active_domains   → [Mathematics]
// decision.firing_rates     → [0.0, 0.87, 0.0]
// decision.input_signals    → DomainSignals { chemistry: 0.0, mathematics: 0.875, .. }

// Reinforce correct routing (potentiates the Mathematics neuron)
router.apply_feedback(VerificationDomain::Mathematics, 1.0);
```

## Sparse Synaptic Map (CSR)

Replace dense $N \times N$ weight matrices with adjacency lists in Compressed Sparse Row format.
For a 2048-neuron network at ~5% connectivity, this reduces storage from ~16 MB to ~800 KB.

```rust
use spikenaut_router::{SparseSynapticMapBuilder, SparseSynapticMap};

const N: usize = 2048;

let map = SparseSynapticMapBuilder::<N>::new()
    .with_self_weight(0.9)
    .with_self_connections()
    .with_lateral_inhibition(-0.15)
    .build();

// Export for GPU kernel launch
let (row_ptr, col_indices, values) = map.to_gpu_arrays();

// Convert from existing dense weights
let dense = [[0.9; N]; N]; // your dense matrix
let sparse = SparseSynapticMap::from_dense(&dense, 0.01);
```

## SAAQ Adaptation-Aware Routing

Route signals using telemetry from the quantization pipeline to avoid adapted (exhausted) neurons:

```rust
use spikenaut_router::{AhlRouter, TelemetrySnapshot};

let mut router = AhlRouter::new();
let mut telemetry = TelemetrySnapshot::new(3);

// Mark Chemistry neuron as heavily adapted (exhausted)
telemetry.adaptation[0] = 0.9;

// Router modulates stimulus away from adapted neurons
let decision = router.route_adaptive("Balance NaOH + HCl", &telemetry);
```

The LIF neuron now tracks adaptation state internally — each spike increases adaptation,
which decays over time via `adaptation_decay`. The `integrate_adaptive()` method applies
the modulation: `stimulus_effective = stimulus · (1 − adaptation)`.

## Ballast-Lab Feedback Loop

Export routing telemetry as CSV for Julia symbolic regression to discover optimal routing policies:

```rust
use spikenaut_router::{AhlRouter, RoutingPolicy};

let mut router = AhlRouter::new();
let decision = router.route("Balance NaOH + HCl");
let telemetry = router.capture_telemetry(42);

// Export CSV for Ballast-Lab ingestion
let csv = router.telemetry_csv(&telemetry, &decision.firing_rates);
// → "step,domain,adaptation,spike_count,quant_error,firing_rate,active\n..."

// Apply a discovered policy for future routing
let policy = RoutingPolicy {
    alpha: 1.2,   // spike count weight
    beta: 0.7,    // adaptation penalty
    gamma: 0.4,   // quantization error penalty
    delta: 0.9,   // base synaptic strength
    threshold: 0.15,
    description: "ballast-lab epoch 47".into(),
};

let decision = router.route_with_policy("Balance NaOH + HCl", &telemetry, &policy);
```

**The workflow:**
1. `corinth-canal` logs routing decisions and spike sparsity to `snn_gpu_routing_telemetry.csv`
2. `ballast-lab` (Julia) performs symbolic regression on that telemetry
3. Julia discovers an optimal routing policy equation
4. Update the `RoutingPolicy` parameters with the discovered coefficients

## Routing Model

Each domain neuron integrates keyword-density features from the input:

```
V_i[t+1] = V_i[t] · (1 − β) + Σ_j W_ij · x_j[t] · (1 − adaptation_i) − Σ_k≠i W_inh · V_k[t]
```

Fires when `V_i ≥ θ_i`. STDP reward: `ΔW += η·(1−W)` on correct fire, `ΔW -= η·W` on miss.
Adaptation increases on each spike and decays exponentially: `adaptation ← adaptation · (1 − decay)`.

*Bi & Poo (1998); Maass (2000) — winner-take-all; Lapicque (1907); Stein (1967)*

## Extending to Custom Domains

```rust
use spikenaut_router::{AhlRouter, VerificationDomain};

// Add your own domain by implementing VerificationDomain
// and providing keyword sets for feature extraction
```

## Extracted from Production

Extracted from [Eagle-Lander](https://github.com/rmems/Eagle-Lander), where it served
as the Anti-Hallucination Layer (AHL) routing verification queries to the correct
neural lobe. Decoupled from lobe-specific logic so it works as a general sparse router
for any classification task.

## Part of the Spikenaut Ecosystem

| Library | Purpose |
|---------|---------|
| [spikenaut-encoder](https://github.com/rmems/spikenaut-encoder) | Feature → spike encoding |
| [spikenaut-backend](https://github.com/rmems/spikenaut-backend) | SNN backend abstraction |

## License

GPL-3.0-or-later
