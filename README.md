# Computational Neuroscience: STDP Simulations

A collection of biologically inspired neuron simulations built using the NEURON simulator and Python.

This project explores Spike-Timing-Dependent Plasticity (STDP) and synaptic learning through progressively more advanced experiments, culminating in a fully animated 3D real-time learning visualization.

---

## Core Experiments

### 1. 3D Animated Online STDP (`stdp_3d_animated.py`)
- Real-time synaptic learning
- 3D neuron visualization
- Live weight updates
- Phase-plane analysis
- Interactive controls (Play, Replay, Speed Slider)

This simulation demonstrates online plasticity where synaptic weights update instantly during spike detection.

---

### 2. Static 3D Morphology + Offline STDP (`stdp_3d_lab.py`)
- 3D neuron morphology via `pt3dadd`
- Offline weight calculation
- Composite visualization (3D + voltage traces + weight bar)

---

### 3. Interactive CLI Firing Experiments (`first_neuron.py`)
- Terminal-based current injection experiments
- Multi-trial simulation loops
- Automatic timestamped plot saving

---

### 4. Foundational STDP Demonstration (`stdp_lab.py`)
- Two-neuron network
- Exponential synapse
- Long-Term Potentiation (LTP) demonstration
- Weight update rule: A₊ e^(−Δt/τ)

---

## Technologies Used
- NEURON simulator
- Python
- NumPy
- Matplotlib
- TkAgg backend (macOS optimized)

---

## Installation

```bash
pip install -r requirements.txt

Note: Requires NEURON installed locally.
