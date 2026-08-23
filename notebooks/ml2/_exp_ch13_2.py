"""Single-shot numeric verification for ML2 13.2 (run once, adopt results)."""
import time
import numpy as np
import gymnasium as gym
import inspect
import gymnasium.envs.classic_control.pendulum as PendMod

def run_ep(env, policy, n_steps, seed):
    s, _ = env.reset(seed=seed)
    tot = 0.0
    n = 0
    for _ in range(n_steps):
        a = policy(s)
        s, r, term, trunc, info = env.step(a)
        tot += r
        n += 1
        if term or trunc:
            break
    return tot, n

print("=" * 60)
print("0) Pendulum reset_model source (start angle?)")
src = inspect.getsource(PendMod.PendulumEnv)
i = src.find("def reset_model")
print(src[i:i+350])

print("=" * 60)
print("1) Pendulum-v1: random vs PD (same code as section original)")
pend = gym.make("Pendulum-v1")
for seed in [0, 1, 2, 3]:
    r, L = run_ep(pend, (lambda e: lambda s: e.action_space.sample())(pend), 50, seed)
    print(f"   random 50-step seed{seed}: {r:.1f}")

def pd_policy(kp, kd):
    def pol(s):
        theta = np.arctan2(s[1], s[0])
        return np.array([np.clip(-kp * theta - kd * s[2], -2.0, 2.0)])
    return pol

for seed in [0, 1, 2, 3]:
    r, L = run_ep(pend, pd_policy(2.0, 0.5), 50, seed)
    print(f"   PD(2,0.5) 50-step seed{seed}: {r:.1f}")
pend.close()

print("=" * 60)
print("2) Pendulum PD grid: kp in {1,2,4,8} x kd in {0,0.5,1,2,4}, 200 steps, 3 seeds mean")
pend = gym.make("Pendulum-v1")
grid = {}
for kp in [1, 2, 4, 8]:
    row = []
    for kd in [0, 0.5, 1, 2, 4]:
        rets = [run_ep(pend, pd_policy(kp, kd), 200, s)[0] for s in [0, 1, 2]]
        grid[(kp, kd)] = float(np.mean(rets))
        row.append(f"{grid[(kp,kd)]:9.1f}")
    print(f"   kp={kp:<2}: " + "  ".join(row))
best = max(grid, key=grid.get)
print("   best:", best, round(grid[best], 1))

def trajectory(kp, kd, steps=200, seed=0):
    env = gym.make("Pendulum-v1")
    s, _ = env.reset(seed=seed)
    th, thd = [], []
    for _ in range(steps):
        theta = np.arctan2(s[1], s[0])
        th.append(theta); thd.append(s[2])
        a = np.array([np.clip(-kp * theta - kd * s[2], -2.0, 2.0)])
        s, r, t, tr, i = env.step(a)
    env.close()
    return np.array(th), np.array(thd)
t1, d1 = trajectory(8, 1.0)
t2, d2 = trajectory(4, 1.0)
print("   traj kp=8,kd=1: final (th,thdot)=", round(float(t1[-1]),3), round(float(d1[-1]),3),
      " tail|th| min:", round(float(np.abs(t1[-30:]).min()),3), " tail|thd| max:", round(float(np.abs(d1[-30:]).max()),3))
print("   traj kp=4,kd=1: final (th,thdot)=", round(float(t2[-1]),3), round(float(d2[-1]),3),
      " tail th range:", round(float(t2[-30:].min()),3), round(float(t2[-30:].max()),3),
      " tail|thd| max:", round(float(np.abs(d2[-30:]).max()),3))

print("=" * 60)
print("3) Reacher-v5: random + info keys + goal radius")
reach = gym.make("Reacher-v5")
for seed in [0, 1, 2]:
    r, L = run_ep(reach, (lambda e: lambda s: e.action_space.sample())(reach), 100, seed)
    print(f"   random 100-step seed{seed}: {r:.2f} ({L} steps)")
s, _ = reach.reset(seed=0)
a = reach.action_space.sample()
s, r, term, trunc, info = reach.step(a)
print("   info keys:", sorted(info.keys()))
print("   info sample:", {k: (round(float(v), 3) if np.isscalar(v) else np.round(v, 3).tolist()) for k, v in info.items()})
import gymnasium.envs.mujoco.reacher_v5 as RMod
rsrc = inspect.getsource(RMod)
i = rsrc.find("while True")
print("   goal sampling src:", rsrc[i:i+230].replace("\n", " | ")[:230])
rad = []
for seed in range(100):
    s, _ = reach.reset(seed=seed)
    rad.append(np.hypot(s[4], s[5]))
rad = np.array(rad)
print(f"   goal radius: mean={rad.mean():.3f} (theory 2R/3={2*0.2/3:.3f}), max={rad.max():.3f}")
reach.close()

print("=" * 60)
print("4) Ant-v5: random 100-step, termination, displacement, info")
ant = gym.make("Ant-v5")
for seed in [0, 1, 2]:
    s, _ = ant.reset(seed=seed)
    x0 = s[0]
    tot = 0.0; steps = 0; healthy = 0; rfor = 0.0; rctrl = 0.0; rcont = 0.0; rhealthy = 0.0
    terminated = False
    for _ in range(100):
        a = ant.action_space.sample()
        s, r, term, trunc, info = ant.step(a)
        tot += r; steps += 1
        if "healthy" in info: healthy += info["healthy"]
        if "reward_forward" in info: rfor += info["reward_forward"]
        if "reward_ctrl" in info: rctrl += info["reward_ctrl"]
        if "reward_contact" in info: rcont += info["reward_contact"]
        if "reward healthy" in info: pass
        if "healthy_reward" in info: rhealthy += info["healthy_reward"]
        if term or trunc:
            terminated = True
            break
    print(f"   seed{seed}: ret={tot:8.2f} steps={steps:3d} terminated={terminated} "
          f"healthy_steps={healthy:3d} x0={x0:+.3f} dx={s[0]-x0:+.3f}m "
          f"sum[forward={rfor:+.2f} ctrl={rctrl:.2f} contact={rcont:.2f}]")
s, _ = ant.reset(seed=0)
a = ant.action_space.sample()
s, r, term, trunc, info = ant.step(a)
print("   info keys:", sorted(info.keys()))
print("   info sample:", {k: (round(float(v), 4) if np.isscalar(v) else v) for k, v in info.items()})
asrc = inspect.getsource(__import__("gymnasium.envs.mujoco.ant_v5", fromlist=["x"]))
i = asrc.find("def _is_healthy")
print("   _is_healthy:", asrc[i:i+160].replace("\n", " | ")[:160])
for l in asrc.splitlines():
    if "min_zheight" in l and "=" in l:
        print("   ", l.strip()[:100]); break
ant.close()

print("=" * 60)
print("5) relative step timing (2000 steps each)")
for name in ["Reacher-v5", "Ant-v5"]:
    e = gym.make(name)
    s, _ = e.reset(seed=0)
    t0 = time.time()
    for _ in range(2000):
        s, r, t, tr, i = e.step(e.action_space.sample())
        if t or tr:
            s, _ = e.reset()
    dt = (time.time() - t0) / 2000
    print(f"   {name}: {dt*1e6:.0f} us/step  ({2000*dt:.2f}s total for 2000 steps)")
    e.close()
print("DONE-EXP")
