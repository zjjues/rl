```markdown

\# Stage 3: Long-Run Optimization and Baseline Comparison



You have successfully stabilized the I-MAPPO training pipeline in Stage 2 (Critic loss is bounded, multi-regime evaluation works, and crashes are prevented). However, the current runs (e.g., `episodes=12`, `steps=30`) are just short smoke tests. 



To generate valid convergence curves and ablation data for an academic paper, we must scale up to a \*\*Long-Run Training Regime\*\*, tune the reward balance, and implement \*\*Baseline Algorithms\*\* and \*\*Intent Mutation (Zero-shot Transfer)\*\* tests.



Please update the Python codebase (`uav\_scheduling\_env.py` and `imappo\_experiments.py`) strictly following these 3 structured tasks. \*\*Do NOT break the existing Stage 2 architecture (e.g., keeping tanh-squashed actor, GAE, PPO clip, and tier-evaluations).\*\*



\## Task 1: Reward Reshaping for High-Density Safety

Currently, the policy shows a 60% collision rate in the "Dense" (Hard) evaluation tier and only \~50% task completion. The agent is trapped in a sub-optimal policy because the collision penalty triggers too late, and the task reward is too weak.



Please update the `reward(self)` function in `uav\_scheduling\_env.py`:

1\. \*\*Increase Task Reward:\*\* Multiply the `reward\_task` component by a factor of `2.0` to strongly encourage target tracking.

2\. \*\*Smooth Safety Shaping (Pre-collision Penalty):\*\* Instead of only penalizing when a hard collision occurs, add a continuous repulsive force `reward\_safety`.

&#x20;  - \*Logic:\* For any distance `d\_ij` between agents `i` and `j`, if `d\_ij < 2.0 \* D\_safe`, apply a soft penalty: `penalty = - k \* (2.0 \* D\_safe - d\_ij)`. 

&#x20;  - This "repulsive potential field" warns the Actor \*before\* the hard collision boundary is breached.

3\. Keep the total `R\_env` clamped within `\[-2.0, 2.0]` to protect the Critic.



\## Task 2: Implement Baselines and Scaling up `imappo\_experiments.py`

We need to generate comparative convergence curves (I-MAPPO vs. MAPPO).

Please update `imappo\_experiments.py`:

1\. \*\*Add Baseline Support:\*\* Add an argument `--algorithm` which accepts `\['imappo', 'mappo']`.

&#x20;  - If `algorithm == 'mappo'`, explicitly disable the action masking (`M = all ones`) and set `eta = 0.0` (disable potential reward shaping). 

&#x20;  - Also, for 'mappo', bypass the Cross-Attention module in the Critic and use a simple concatenation of states/observations (or set attention weights to uniform).

2\. \*\*Scale up Default Parameters:\*\* Change the default argparse values to support a long run:

&#x20;  - `episodes = 3000`

&#x20;  - `steps = 50`

&#x20;  - `rollout = 128` (Increase rollout length for better advantage estimation)

&#x20;  - `batch\_size = 64`

&#x20;  - `eval\_interval = 100` (Evaluate less frequently to speed up training)

3\. \*\*Logging updates:\*\* Ensure that `episode\_reward\_env`, `episode\_collisions`, and `episode\_task\_completion` are smoothly logged via TensorBoard or saved continuously to a CSV/JSON file over the 3000 episodes so we can plot the convergence curve.



\## Task 3: The "Intent Mutation" Evaluation Script

To prove the zero-shot policy transfer capability of the Cross-Attention Critic, we need a standalone evaluation script: `test\_intent\_mutation.py`.



Please write this new script with the following logic:

1\. \*\*Load Checkpoint:\*\* Load the trained weights of a successful `I-MAPPO` model.

2\. \*\*Setup Environment:\*\* Initialize the environment with 4 UAVs and 3 targets.

3\. \*\*Phase 1 (Approaching):\*\* For `step = 0` to `step = 20`, feed the Actor/Critic a fixed Intent vector `I\_1 = \[1, 0, 0, ...]` (representing "Gather/Attack"). Record the positions of all UAVs.

4\. \*\*Phase 2 (Emergency Evasion):\*\* At `step = 21`, forcefully flip the Intent vector to `I\_2 = \[0, 1, 0, ...]` (representing "Disperse/Evade"). Update the Action Mask `M` accordingly.

5\. \*\*Metric Calculation:\*\* Track the UAVs' velocities and coordinates after `step = 21`. 

&#x20;  - Calculate the \*\*"Response Latency"\*\*: How many steps it takes for all UAVs to reverse their velocity vectors away from the targets.

&#x20;  - Print the Latency and save the trajectory coordinates `(x, y)` over time into a JSON file (e.g., `mutation\_trajectory.json`) so we can plot it later.



Please provide the specific code modifications for `uav\_scheduling\_env.py` (Task 1), `imappo\_experiments.py` (Task 2), and the complete code for `test\_intent\_mutation.py` (Task 3). Maintain clean, modular Python code compatible with PyTorch and PettingZoo.

```



\---

