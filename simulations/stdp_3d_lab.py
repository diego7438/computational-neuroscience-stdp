"""
NEURON LAB: 3D Learning & Memory with STDP
Author: Diego (17 y/o scientist)
Purpose: Demonstrate biological learning using STDP with 3D neuron visualization
"""

from neuron import h
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Create neurons
# ----------------------------
def create_neuron(name):
    sec = h.Section(name=name)
    sec.L = sec.diam = 12.6
    sec.insert('hh')
    return sec

pre = create_neuron("pre")
post = create_neuron("post")

# Connect post to pre (simple one-way synapse)
syn = h.ExpSyn(post(0.5))
syn.tau = 2

# NetCon: pre spike -> synapse
nc = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
nc.weight[0] = 0.01  # initial synaptic weight

# ----------------------------
# STDP parameters
# ----------------------------
A_plus = 0.005
A_minus = -0.005
tau_stdp = 20  # ms

# Record spike times
pre_spikes = h.Vector()
post_spikes = h.Vector()

nc_pre_spike = h.NetCon(pre(0.5)._ref_v, None, sec=pre)
nc_pre_spike.threshold = 0  # spike threshold
nc_pre_spike.record(pre_spikes)

nc_post_spike = h.NetCon(post(0.5)._ref_v, None, sec=post)
nc_post_spike.threshold = 0
nc_post_spike.record(post_spikes)

# ----------------------------
# Stimulate neurons
# ----------------------------
stim_pre = h.IClamp(pre(0.5))
stim_pre.delay = 5
stim_pre.dur = 1
stim_pre.amp = 0.8

stim_post = h.IClamp(post(0.5))
stim_post.delay = 15  # post fires after pre
stim_post.dur = 1
stim_post.amp = 0.8

# ----------------------------
# Record voltage and time
# ----------------------------
t_vec = h.Vector().record(h._ref_t)
v_pre = h.Vector().record(pre(0.5)._ref_v)
v_post = h.Vector().record(post(0.5)._ref_v)

# ----------------------------
# Simulation
# ----------------------------
h.finitialize(-65)
tstop = 50  # ms

while h.t < tstop:
    h.fadvance()

# ----------------------------
# STDP update
# ----------------------------
final_weight = nc.weight[0]
for t_pre in pre_spikes:
    for t_post in post_spikes:
        dt = t_post - t_pre
        if dt > 0:
            final_weight += A_plus * np.exp(-dt / tau_stdp)
        else:
            final_weight += A_minus * np.exp(dt / tau_stdp)

# ----------------------------
# 3D neuron visualization (Matplotlib)
# ----------------------------
def plot_3d_neuron(sections, title="Neuron Morphology"):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    for sec in sections:
        xs, ys, zs = [], [], []
        h.pt3dclear(sec=sec)
        # Simple straight cylinder: soma at origin, dendrite offset
        if sec == pre:
            h.pt3dadd(0,0,0,12.6, sec=sec)
        else:
            h.pt3dadd(0,0,0,12.6, sec=sec)
            h.pt3dadd(50,20,10,2, sec=sec)
        for i in range(int(h.n3d(sec=sec))):
            xs.append(h.x3d(i, sec=sec))
            ys.append(h.y3d(i, sec=sec))
            zs.append(h.z3d(i, sec=sec))
        ax.plot(xs, ys, zs, '-o', label=sec.name())
    ax.set_title(title)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    ax.legend()
    plt.show()

plot_3d_neuron([pre, post], title="3D Neurons with Synapse")

# ----------------------------
# Plot voltages and synaptic weight
# ----------------------------
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(t_vec, v_pre, label="Pre-synaptic")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t_vec, v_post, label="Post-synaptic", color="orange")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
plt.bar([0], [final_weight], width=0.5)
plt.ylabel("Synaptic Weight")
plt.xticks([0], ["Pre → Post"])
plt.grid(True)

plt.xlabel("Time (ms)")
plt.suptitle("STDP Learning Lab")
plt.tight_layout()
plt.show()

print("Initial weight:", nc.weight[0])
print("Final weight after STDP:", final_weight)
