# Chapter 14. Advanced Simulation: MuJoCo & Isaac Sim

The basic Gymnasium environments used in Chapter 13 compute physics using
simplified approximations — fast to train with, but physical phenomena
that matter for robot control, like contact and friction, are only
roughly imitated. Real robotics research uses far more sophisticated
physics engines. This chapter introduces two of them — **MuJoCo**, a
precise physics engine that runs even on CPU, and **NVIDIA Isaac Sim**,
which runs thousands of simulations at once on GPU — and lays out when to
use which.

## 14.1 MuJoCo: Precise Contact and Friction Simulation

**MuJoCo** (Multi-Joint dynamics with Contact) is a physics engine
designed to compute a robot's joints, contacts, and friction with
physical accuracy — originally commercial software, it was released free
in 2021 and has since become the de facto standard tool for robotics
reinforcement learning research. A MuJoCo model defines an object's shape,
joints, mass, and so on in XML, and simulation proceeds by repeatedly
stepping this model forward:

```python
import mujoco

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print("Position degrees of freedom (nq):", model.nq, "Velocity degrees of freedom (nv):", model.nv)

for step in range(5):
    mujoco.mj_step(model, data)  # advance one step according to physical laws like gravity
print("Sphere height after 5 steps:", round(data.qpos[2], 4))  # drops slightly from free fall
```

This code is the simplest possible example — a single ball falling under
gravity — but the same API (`MjModel`, `MjData`, `mj_step`) can represent
multi-legged robots, robot arms, and even humanoid robots. Gymnasium's
`HalfCheetah`, `Ant`, and `Humanoid` continuous-control environments are
actually all built on top of this MuJoCo engine. The DQN and PPO
algorithms covered in Chapters 9-13 don't change whether the environment
is CartPole or a MuJoCo-based robot — as long as the `env.step()`
interface is respected, the algorithm-side code stays the same no matter
how much more sophisticated the physics engine gets.

## 14.2 NVIDIA Isaac Sim: Thousands at Once on GPU

**Isaac Sim** is a robot simulation environment built on NVIDIA's
Omniverse platform, which performs the physics computation itself in
parallel on the **GPU** — instead of running the same robot environment
one at a time sequentially on CPU, it simulates thousands of copies
simultaneously on the GPU, dramatically speeding up how quickly you can
gather the experience needed for training. It also supports realistic
rendering (including camera sensor simulation), so it's widely used in
research on policies that "decide by looking at pixels."

**This semester's exercises do not run Isaac Sim directly** — its minimum
requirement is an RTX 4080-class GPU (16GB VRAM), and it simply doesn't
support datacenter GPUs without RT cores (the A100, H100 class) at all.
The GPUs the school owns are exactly this H100/H200 class, so Isaac Sim
can't be used for hands-on exercises — this chapter covers it only through
principles and demo footage.

## 14.3 The Principle Behind GPU-Accelerated Simulation

CPU-based simulation (MuJoCo, Gymnasium's default environments) steps one
environment at a time — gathering more experience simply takes more time.
A GPU-accelerated simulator like Isaac Sim redesigns the physics
computation itself as large-scale parallel computation — similar in spirit
to how ML1's CNN (Section 10.4) reuses "the same operation, repeated
across positions," except now it's "the same physics step, repeated across
robot copies," scattering thousands of environments across GPU cores to
compute simultaneously. As a result, the amount of (state, action, reward)
experience gathered per second can be tens to hundreds of times higher
than on CPU — Chapter 9's experience replay buffer would fill up almost
instantly.

## 14.4 Which Tool to Use When

| Tool | Runs on | Precision | Speed (large-scale parallel) | Use this semester |
|---|---|---|---|---|
| Gymnasium default envs | CPU | Low-medium | Slow | Basic exercises |
| PyBullet | CPU | Medium | Slow | Robot arm/quadruped exercises |
| MuJoCo | CPU | High | Slow | Precision control exercises |
| NVIDIA Isaac Sim | GPU | High | Very fast (thousands-fold parallel) | Demo only (hardware constraint) |

Summarizing the decision criteria: **if you want to run it directly on
your own laptop**, use Gymnasium/PyBullet; **if you need precise contact
and friction matching**, use MuJoCo; **if you need to gather a huge amount
of experience very quickly** (and have the right GPU), use Isaac Sim. All
three share the same property — you're just "swapping in a different
environment" for the algorithms covered in Chapters 9-13.

**The reinforcement learning algorithm itself doesn't change no matter
which simulation tool you use — what changes is only the tradeoff between
physical accuracy and computational speed. Which tool to use is decided by
answering two questions: "how precise does this need to be?" and "how much
experience do I actually need?"**

---

## Exercises

**1. (Hands-on)** Run the MuJoCo code above as-is, then change the ball's
initial height in the `xml` (`pos="0 0 1"`) to `pos="0 0 5"` and re-run,
comparing how the height after 5 steps differs. Increase the number of
steps to 50 and check whether the ball comes to rest after hitting the
floor (`plane`).

**2. (Conceptual)** Compare, in two or three sentences, how GPU-accelerated
simulation "repeating the same computation across many copies" is similar
to how ML1's CNN "applies the same filter repeatedly across many
positions" — and what's different (is what's being shared "parameters" or
"environment copies"?).

**3. (Conceptual)** If the school were to acquire a GPU suitable for Isaac
Sim exercises (RTX 4080/16GB class or better), pick which exercise from
this semester's curriculum (Chapters 9-13) would benefit the most from
being moved to Isaac Sim, and explain why (hint: think about which
algorithm needs to gather the most experience).
