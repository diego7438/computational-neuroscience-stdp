"""
NEURON LAB: 3D STDP Animation (Mac-friendly, voltage-clamped)
Author: Diego (17 y/o scientist)
Purpose: Animate pre/post neuron voltage and synapse learning in 3D
"""

from neuron import h
import matplotlib
# Force TkAgg backend to prevent Mac crashes and timer issues
try:
    matplotlib.use('TkAgg')
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, Slider
import numpy as np
import os

# Try to import Rich for pretty terminal output
try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    print("To make this project spiffier, run: pip install rich")
    import sys; sys.exit()

console.print("[bold green]--> Initializing NEURON simulation...[/bold green]")
console.print(f"[dim]Matplotlib Backend: {matplotlib.get_backend()}[/dim]")

# ----------------------------
# Create neurons with 3D structure
# ----------------------------
def create_neuron(name, start_pos=(0,0,0), end_pos=(50,20,10)):
    sec = h.Section(name=name)
    sec.L = sec.diam = 12.6
    sec.insert('hh')
    # 3D morphology
    h.pt3dclear(sec=sec)
    h.pt3dadd(*start_pos, sec.diam, sec=sec)
    h.pt3dadd(*end_pos, 2, sec=sec)
    return sec

# Both neurons now have visible dendrites
pre = create_neuron("pre", start_pos=(0,0,0), end_pos=(20,0,10))
post = create_neuron("post", start_pos=(0,0,0), end_pos=(50,20,10))

# ----------------------------
# Synapse and NetCon
# ----------------------------
syn = h.ExpSyn(post(0.5))
syn.tau = 2
nc = h.NetCon(pre(0.5)._ref_v, syn, sec=pre)
nc.weight[0] = 0.01  # initial synaptic weight

# ----------------------------
# STDP parameters
# ----------------------------
A_plus = 0.005
A_minus = -0.005
tau_stdp = 20

# Spike recording
pre_spikes = h.Vector()
post_spikes = h.Vector()
nc_pre = h.NetCon(pre(0.5)._ref_v, None, sec=pre)
nc_pre.threshold = 0
nc_pre.record(pre_spikes)
nc_post = h.NetCon(post(0.5)._ref_v, None, sec=post)
nc_post.threshold = 0
nc_post.record(post_spikes)

# ----------------------------
# Stimuli
# ----------------------------
stim_pre = h.IClamp(pre(0.5))
stim_pre.delay = 5
stim_pre.dur = 1
stim_pre.amp = 0.8

stim_post = h.IClamp(post(0.5))
stim_post.delay = 15
stim_post.dur = 1
stim_post.amp = 0.8

# ----------------------------
# Record voltage and time
# ----------------------------
t_vec = h.Vector().record(h._ref_t)
v_pre = h.Vector().record(pre(0.5)._ref_v)
v_post = h.Vector().record(post(0.5)._ref_v)

# ----------------------------
# Simulation setup
# ----------------------------
h.finitialize(-65)
tstop = 50  # ms
steps_per_frame = 10  # Simulate 10 steps per animation frame for speed

# ----------------------------
# Matplotlib 3D setup
# ----------------------------
fig = plt.figure(figsize=(18,6))  # Made wider to fit 3 graphs

ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.set_xlim(-10, 60)
ax.set_ylim(-10, 30)
ax.set_zlim(-10, 30)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Neuron Voltage Animation")
ax.view_init(elev=20, azim=45)

# Pre-define line objects
pre_line, = ax.plot([], [], [], '-o', lw=2, color='blue', label='Pre')
post_line, = ax.plot([], [], [], '-o', lw=2, color='orange', label='Post')
syn_line, = ax.plot([], [], [], 'o-', lw=3, color='purple', label='Synapse')
ax.legend()

# 2D Live Graph setup (Subplot 2)
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_title("Live Synaptic Weight Change")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel('Weight')
ax2.set_xlim(0, tstop)
ax2.set_ylim(0, 0.02)
weight_line_2d, = ax2.plot([], [], 'g-', lw=2, label='Weight')
ax2.grid(True)

# 3D Phase Plane setup (Subplot 3) - CALCULUS IN ACTION
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_title("Phase Plane: V vs dV/dt (The Derivative)")
ax3.set_xlabel("Voltage (V)")
ax3.set_ylabel("Derivative (dV/dt)")
ax3.set_xlim(-80, 50)
ax3.set_ylim(-100, 300)  # Derivatives get huge during spikes!
phase_line, = ax3.plot([], [], 'c-', lw=1.5, alpha=0.8)
ax3.grid(True)

# Text label for live weight
weight_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, color='black')

# Title Screen Text (Centered)
title_text = fig.text(0.5, 0.5, "", ha='center', va='center', fontsize=16, color='black', fontweight='bold')

# ----------------------------
# Clamp voltage for valid RGB
# ----------------------------
def clamp_color(val):
    return max(0, min(1, val))

# ----------------------------
# Animation update function
# ----------------------------
def update(frame):
    # --- Title Screen Logic ---
    title_duration = 100  # frames (approx 3-4 seconds)
    
    if frame < title_duration:
        title_text.set_text("STDP LEARNING LAB\n\n"
                            "Watch the synapse (purple) strengthen\n"
                            "as neurons fire together.\n\n"
                            "Pre-synaptic: Blue  |  Post-synaptic: Orange")
        # Rotate camera gently during title
        try:
            ax.view_init(elev=20, azim=45 + frame * 0.2)
        except AttributeError:
            pass
        return pre_line, post_line, syn_line, weight_text, weight_line_2d, title_text, phase_line

    # --- Simulation Logic ---
    title_text.set_text("")  # Hide title
    sim_frame = frame - title_duration  # Adjust frame count for simulation

    if sim_frame == 0:
        h.finitialize(-65)
        update.last_pre_count = 0
        update.last_post_count = 0
        nc.weight[0] = initial_weight
        time_data.clear()
        weight_data.clear()
        phase_v_data.clear()
        phase_dv_data.clear()
        console.print("[bold magenta]--> Starting animation loop (Reset)...[/bold magenta]")

    # Advance simulation by multiple steps to speed up animation
    for _ in range(steps_per_frame):
        h.fadvance()

    # --- Online STDP Implementation ---
    # Check for new Pre spikes (LTD check: Post fired before Pre)
    while pre_spikes.size() > update.last_pre_count:
        console.print(f"[bold cyan][t={h.t:.1f}ms] ⚡ Pre-synaptic Spike detected![/bold cyan]")
        os.system('afplay /System/Library/Sounds/Tink.aiff &') # Mac sound effect
        t_pre = pre_spikes[update.last_pre_count]
        for t_post in post_spikes:
            dt = t_post - t_pre
            if dt <= 0: 
                old_w = nc.weight[0]
                nc.weight[0] += A_minus * np.exp(dt / tau_stdp)
                console.print(f"    [bold red]-> LTD: Weight weakened ({old_w:.5f} -> {nc.weight[0]:.5f})[/bold red]")
        update.last_pre_count += 1

    # Check for new Post spikes (LTP check: Pre fired before Post)
    while post_spikes.size() > update.last_post_count:
        console.print(f"[bold yellow][t={h.t:.1f}ms] ⚡ Post-synaptic Spike detected![/bold yellow]")
        os.system('afplay /System/Library/Sounds/Tink.aiff &') # Mac sound effect
        t_post = post_spikes[update.last_post_count]
        for t_pre in pre_spikes:
            dt = t_post - t_pre
            if dt > 0:
                old_w = nc.weight[0]
                nc.weight[0] += A_plus * np.exp(-dt / tau_stdp)
                console.print(f"    [bold green]-> LTP: Weight strengthened ({old_w:.5f} -> {nc.weight[0]:.5f})[/bold green]")
        update.last_post_count += 1
    # ----------------------------------

    # Voltage coloring
    # Pre-neuron: Blue (low V) to Cyan (high V)
    pre_color = (0, clamp_color((v_pre[sim_frame]+70)/100), 1)
    # Post-neuron: Orange (low V) to Yellow (high V)
    post_color = (1, clamp_color(0.5 + (v_post[sim_frame]+70)/100), 0)

    # Update pre neuron
    xs, ys, zs = [h.x3d(i, sec=pre) for i in range(int(h.n3d(sec=pre)))], \
                 [h.y3d(i, sec=pre) for i in range(int(h.n3d(sec=pre)))], \
                 [h.z3d(i, sec=pre) for i in range(int(h.n3d(sec=pre)))]
    pre_line.set_data(xs, ys)
    pre_line.set_3d_properties(zs)
    pre_line.set_color(pre_color)

    # Update post neuron
    xs, ys, zs = [h.x3d(i, sec=post) for i in range(int(h.n3d(sec=post)))], \
                 [h.y3d(i, sec=post) for i in range(int(h.n3d(sec=post)))], \
                 [h.z3d(i, sec=post) for i in range(int(h.n3d(sec=post)))]
    post_line.set_data(xs, ys)
    post_line.set_3d_properties(zs)
    post_line.set_color(post_color)

    # Synapse connects end of pre to start of post
    syn_x = [h.x3d(int(h.n3d(sec=pre))-1, sec=pre), h.x3d(0, sec=post)]
    syn_y = [h.y3d(int(h.n3d(sec=pre))-1, sec=pre), h.y3d(0, sec=post)]
    syn_z = [h.z3d(int(h.n3d(sec=pre))-1, sec=pre), h.z3d(0, sec=post)]
    syn_line.set_data(syn_x, syn_y)
    syn_line.set_3d_properties(syn_z)
    syn_line.set_markersize(10 + nc.weight[0]*200)  # synapse grows as weight increases

    # Rotate the camera (azim changes with frame count)
    try:
        ax.view_init(elev=20, azim=45 + frame * 0.5)
    except AttributeError:
        pass

    # Update 2D graph
    time_data.append(h.t)
    weight_data.append(nc.weight[0])
    weight_line_2d.set_data(time_data, weight_data)

    weight_text.set_text(f"Synaptic Weight: {nc.weight[0]:.5f}")

    # Update Phase Plane (Calculus!)
    # Calculate derivative: (Current V - Previous V) / dt
    if len(v_pre) > 1:
        # Get the last few points to draw a trail
        current_v = v_pre[-1]
        prev_v = v_pre[-2]
        dv_dt = (current_v - prev_v) / h.dt
        phase_v_data.append(current_v)
        phase_dv_data.append(dv_dt)
        phase_line.set_data(phase_v_data, phase_dv_data)

    return pre_line, post_line, syn_line, weight_text, weight_line_2d, title_text, phase_line

# ----------------------------
# Run animation
# ----------------------------
# Initialize counters and capture initial weight
update.last_pre_count = 0
update.last_post_count = 0
initial_weight = nc.weight[0]
time_data = []
weight_data = []
phase_v_data = []
phase_dv_data = []

# Add extra frames for the title screen
total_frames = 100 + int(tstop / (h.dt * steps_per_frame))
anim = FuncAnimation(fig, update, frames=total_frames, interval=30, blit=False, repeat=False)

# To save as MP4, uncomment the line below (requires ffmpeg installed)
# try:
#     print("Saving video to stdp_learning.mp4... (this may take a moment)")
#     anim.save('stdp_learning.mp4', writer='ffmpeg', fps = 30)
#     print(f"Video saved successfully at: {os.path.abspath('stdp_learning.mp4')}")
# except Exception as e:
#     print(f"Could not save video (ffmpeg might be missing): {e}")

# Start the animation in a paused state
anim.pause()

# --- UI WIDGETS ---

# CheckButtons for toggling lines
ax_check = plt.axes([0.02, 0.4, 0.1, 0.15])
check = CheckButtons(ax_check, ('Pre', 'Post', 'Synapse'), (True, True, True))

def toggle_lines(label):
    if label == 'Pre':
        pre_line.set_visible(not pre_line.get_visible())
    elif label == 'Post':
        post_line.set_visible(not post_line.get_visible())
    elif label == 'Synapse':
        syn_line.set_visible(not syn_line.get_visible())
    plt.draw()

check.on_clicked(toggle_lines)

# Playback and Speed Controls
ax_play = plt.axes([0.65, 0.05, 0.1, 0.075])
btn_play = Button(ax_play, 'Start', color='0.9', hovercolor='0.95')
btn_play.label.set_color('black')

ax_replay = plt.axes([0.81, 0.05, 0.1, 0.075])
btn_replay = Button(ax_replay, 'Replay', color='0.9', hovercolor='0.95')
btn_replay.label.set_color('black')

is_playing = False

def toggle_anim(event):
    global is_playing
    if is_playing:
        anim.pause()
        btn_play.label.set_text("Start")
        is_playing = False
    else:
        anim.resume()
        btn_play.label.set_text("Stop")
        is_playing = True

def replay_anim(event):
    global is_playing
    anim.frame_seq = anim.new_frame_seq()
    anim.resume()
    btn_play.label.set_text("Stop")
    is_playing = True

btn_play.on_clicked(toggle_anim)
btn_replay.on_clicked(replay_anim)

# Speed control slider
ax_slider = plt.axes([0.15, 0.02, 0.5, 0.03]) # Made smaller to fit
speed_slider = Slider(ax=ax_slider, label='Speed (ms/frame)', valmin=10, valmax=200, valinit=30)

def update_speed(val):
    anim.event_source.interval = val
speed_slider.on_changed(update_speed)

console.print("[bold blue]--> Opening visualization window...[/bold blue]")
plt.show(block=True)  # keeps window open

# ----------------------------
# Plot voltages + final weight
# ----------------------------
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(t_vec, v_pre, label="Pre-synaptic")
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(t_vec, v_post, label="Post-synaptic", color='orange')
plt.ylabel("Voltage (mV)")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.bar([0], [nc.weight[0]], width=0.5)
plt.ylabel("Synaptic Weight")
plt.xticks([0], ["Pre → Post"])
plt.grid(True)
plt.xlabel("Time (ms)")
plt.suptitle("STDP Learning Lab")

plt.tight_layout()
plt.show()

# Create a summary table
table = Table(title="STDP Learning Results")
table.add_column("Metric", style="cyan", no_wrap=True)
table.add_column("Value", style="magenta")
table.add_row("Initial Weight", f"{initial_weight:.5f}")
table.add_row("Final Weight", f"{nc.weight[0]:.5f}")
console.print(table)