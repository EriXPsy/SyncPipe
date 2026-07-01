"""
Quick runner for morphology_shape_analysis on all 3 datasets.
Saves outputs to artifacts/morphology/morphology_shape_out/.
"""
import sys, json, numpy as np, pandas as pd, os, warnings
from pathlib import Path

# Prevent sklearn from spawning threads (sandbox safety)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.morphology.morphology_shape_analysis import (
    load_traces, method1_traceshape, method2_episodes
)

OUT = Path(__file__).resolve().parents[2] / 'artifacts' / 'morphology' / 'morphology_shape_out'
OUT.mkdir(parents=True, exist_ok=True)

# Step 1: Load
by = load_traces([
    str(Path(__file__).resolve().parents[2] / 'artifacts' / 'wcc_traces' / name)
    for name in ['lerique_wcc_traces.csv', 'gordon_wcc_traces.csv', 'andersen_wcc_traces.csv']
])
print(f'Datasets: {list(by.keys())}')
for ds, tl in by.items():
    print(f'  {ds}: {len(tl)} traces')

# Step 2: Method 1 — scale-free shape clustering
print('\n=== Method 1: Scale-free shape clustering ===')
m1_desc, m1_k = method1_traceshape(by)
m1_desc.to_csv(OUT / 'method1_scalefree_descriptors.csv', index=False)
m1_k.to_csv(OUT / 'method1_k_selection.csv', index=False)
print(f'{len(m1_desc)} descriptors across {m1_desc["dataset"].nunique()} datasets')
print(m1_k.round(3).to_string())

# Step 3: Method 2 — episode waveform archetypes
for mode in ('fixed', 'percentile'):
    print(f'\n=== Method 2 ({mode} threshold) ===')
    try:
        res = method2_episodes(by, mode)
        if res is None:
            print(f'  No episodes extracted')
            continue
        print(f'  {res["n_episodes"]} total episodes')
        print(f'  Waveform k = {res["waveform_k_chosen"]}, Feature k = {res["feature_k_chosen"]}')
        print(f'  Waveform-vs-Feature ARI agreement = {res["wave_vs_feat_ari"]:.3f}')
        print(f'  Waveform k-selection:')
        print(res['waveform_k'].round(3).to_string())
        print(f'  Feature k-selection:')
        print(res['feature_k'].round(3).to_string())

        # Save outputs
        np.savetxt(OUT / f'method2_{mode}_waveform_archetypes.csv',
                   res['waveform_archetypes'], delimiter=',')
        res['feats'].to_csv(OUT / f'method2_{mode}_episode_features.csv', index=False)
        res['waveform_k'].to_csv(OUT / f'method2_{mode}_waveform_k.csv', index=False)
        res['feature_k'].to_csv(OUT / f'method2_{mode}_feature_k.csv', index=False)
        if 'feature_profiles' in res:
            res['feature_profiles'].to_csv(OUT / f'method2_{mode}_cluster_profiles.csv')
            print(f'\n  Cluster shape-feature profiles:')
            print(res['feature_profiles'].round(3).to_string())
    except Exception as e:
        print(f'  ERROR: {e}')
        import traceback; traceback.print_exc()

print(f'\nDone -> {OUT}')
