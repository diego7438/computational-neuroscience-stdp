"""
NEURON LAB: Learning & Memory with STDP
Author: Diego (17 y/o neuroscientist-in-training)
Purpose: Demonstrate biological learning via spike-timing-dependent plasticity
"""

from neuron import h
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Create neurons
# ----------------------------
def create_neuron(name):
    soma = h.Section(name=name)
    soma.L = soma.diam = 12.6
    soma.insert('hh')
    return soma

pre = create_neuron("pre")
post = create_neuron("post")

# ----------------------------
# Synapse with STDP
# ----------------------------
syn = h.ExpSyn(post(0.5))
syn.tau = 2

# NetCon: pre spike -> synapse
nc = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
nc.weight[0] = 0.01   # initial synaptic strength

# STDP parameters
A_plus = 0.005
A_minus = -0.005
tau = 20

# Record spike times
pre_spikes = h.Vector()
post_spikes = h.Vector()

nc_pre_spike = h.NetCon(pre(0.5)._ref_v, None, sec=pre)
nc_pre_spike.record(pre_spikes)

nc_post_spike = h.NetCon(post(0.5)._ref_v, None, sec=post)
nc_post_spike.record(post_spikes)

# ----------------------------
# Stimulate neurons
# ----------------------------
stim_pre = h.IClamp(pre(0.5))
stim_pre.delay = 5
stim_pre.dur = 1
stim_pre.amp = 0.8

stim_post = h.IClamp(post(0.5))
stim_post.delay = 15   # post fires AFTER pre (learning!)
stim_post.dur = 1
stim_post.amp = 0.8

# ----------------------------
# Recording
# ----------------------------
t = h.Vector().record(h._ref_t)
v_pre = h.Vector().record(pre(0.5)._ref_v)
v_post = h.Vector().record(post(0.5)._ref_v)
weights = []

# ----------------------------
# Simulation
# ----------------------------
h.finitialize(-65)
tstop = 50
while h.t < tstop:
    h.fadvance()


# ----------------------------
# STDP update
# ----------------------------
for t_pre in pre_spikes:
    for t_post in post_spikes:
        dt = t_post - t_pre
        if dt > 0:
            nc.weight[0] += A_plus * np.exp(-dt / tau)
        else:
            nc.weight[0] += A_minus * np.exp(dt / tau)

weights.append(nc.weight[0])

# ----------------------------
# Plot results
# ----------------------------
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.plot(t, v_pre)
plt.title("Pre-synaptic Neuron")

plt.subplot(3,1,2)
plt.plot(t, v_post)
plt.title("Post-synaptic Neuron")

plt.subplot(3,1,3)
plt.bar([0], weights)
plt.title("Synaptic Weight After Learning")

plt.tight_layout()
plt.show()

print("Final synaptic weight:", nc.weight[0])
