"""
NEURON LAB: Multi-Trial Interactive Simulation with Auto-Save
Author: Diego (17 y/o scientist)
Purpose: Simulate multiple neurons, experiment interactively,
and save plots automatically for GitHub.
"""

from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# ----------------------
# Create a single neuron
# ----------------------
def create_neuron(name="soma"):
    sec = h.Section(name=name)
    sec.L = sec.diam = 12.6157
    sec.insert('hh')
    return sec

# ----------------------
# Simulation settings
# ----------------------
num_neurons = 3              # number of neurons
neurons = [create_neuron(f"soma{i}") for i in range(num_neurons)]
currents = [0.1 for _ in range(num_neurons)]  # default current per neuron
delay = 5
duration = 40
steps = 5000
num_trials = 3               # number of trials per simulation

# Ensure a folder exists to save plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# ----------------------
# Function to run a simulation and plot
# ----------------------
def run_simulation():
    plt.figure(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0,1,num_neurons*num_trials))

    for neuron_idx, soma in enumerate(neurons):
        for trial in range(num_trials):
            # Inject current
            stim = h.IClamp(soma(0.5))
            stim.delay = delay
            stim.dur = duration
            stim.amp = currents[neuron_idx]

            # Record voltage and time
            t = h.Vector()
            v = h.Vector()
            t.record(h._ref_t)
            v.record(soma(0.5)._ref_v)

            # Initialize and run
            h.finitialize(-65)
            for _ in range(steps):
                h.fadvance()

            # Plot neuron voltage
            plt.plot(t, v, label=f"Neuron {neuron_idx+1} Trial {trial+1}",
                     color=colors[neuron_idx*num_trials + trial])

    # Plot styling
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Multi-Trial Neuron Simulation")
    plt.legend()
    plt.grid(True)

    # Save plot automatically
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/neuron_plot_{timestamp}.png"
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved as {filename}")

# ----------------------
# Interactive loop
# ----------------------
print("Welcome to the Advanced NEURON Lab! ⚡")
print("You have 3 neurons. Type a number to set current (nA) for each neuron, 'run' to simulate, or 'q' to quit.")

while True:
    for i in range(num_neurons):
        ans = input(f"Neuron {i+1} current (nA, current={currents[i]}): ")
        if ans.lower() == 'q':
            exit()
        elif ans.lower() == 'run':
            break
        else:
            try:
                currents[i] = float(ans)
            except Exception:
                print("Invalid input, keeping previous value.")
    run_simulation()
