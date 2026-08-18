import json, glob
import numpy as np
from pathlib import Path

results = {}
for v in ['imappo', 'mappo', 'matd3', 'ippo']:
    results[v] = {'easy': {'c': [], 't': []}, 'medium': {'c': [], 't': []}, 'hard': {'c': [], 't': []}}

for path in glob.glob('experiments/pilot/uav_imappo_main/*/seed_*/result.json'):
    p = Path(path)
    v = p.parent.parent.name
    if v not in results:
        continue
    d = json.loads(p.read_text())
    tm = d['tier_metrics']
    for tier in ['easy', 'medium', 'hard']:
        results[v][tier]['c'].append(tm[tier][f'{tier}_collision_rate'])
        results[v][tier]['t'].append(tm[tier][f'{tier}_task_completion'])

print(f"{'Variant':<8} {'Tier':<7} {'Collision':>10} {'Task':>10}")
print("-" * 40)
for v in ['imappo', 'mappo', 'matd3', 'ippo']:
    for tier in ['easy', 'medium', 'hard']:
        c = results[v][tier]['c']
        t = results[v][tier]['t']
        print(f"{v:<8} {tier:<7} {np.mean(c):>10.4f} {np.mean(t):>10.4f}")
