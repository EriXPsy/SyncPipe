"""
SyncPipe cross-dataset feature evidence matrix.
Ranks 8 features across 4 datasets for paradigm-specific recommendations.
"""
import sys
sys.path.insert(0, r'<REPO>')

features = ['onset_latency','rise_time','peak_amplitude','recovery_time',
            'dwell_time','switching_rate','mean_synchrony','synchrony_entropy']

evidence_bizzego = {
    'onset_latency':   (0, 0, 0),
    'rise_time':       (0, 0, 0),
    'peak_amplitude':  (1, 2, 2),
    'recovery_time':   (0, 0, 0),
    'dwell_time':      (0, 1, 2),
    'switching_rate':  (1, 2, 2),
    'mean_synchrony':  (0, 2, 2),
    'synchrony_entropy':(0, 2, 1),
}

evidence_lerique = {
    'onset_latency':   (0, 0, 0),
    'rise_time':       (0, 0, 0),
    'peak_amplitude':  (2, 2, 2),
    'recovery_time':   (0, 0, 0),
    'dwell_time':      (0, 0, 0),
    'switching_rate':  (2, 2, 2),
    'mean_synchrony':  (1, 1, 0),
    'synchrony_entropy':(0, 0, 0),
}

evidence_gordon = {
    'onset_latency':   (0, 0, 0),
    'rise_time':       (0, 0, 0),
    'peak_amplitude':  (2, 0, 0),
    'recovery_time':   (0, 0, 0),
    'dwell_time':      (0, 0, 0),
    'switching_rate':  (0, 0, 0),
    'mean_synchrony':  (1, 0, 0),
    'synchrony_entropy':(0, 0, 0),
}

evidence_andersen = {
    'onset_latency':   (0, 0, 0),
    'rise_time':       (0, 0, 0),
    'peak_amplitude':  (1, 0, 0),
    'recovery_time':   (0, 0, 0),
    'dwell_time':      (0, 0, 0),
    'switching_rate':  (0, 0, 0),
    'mean_synchrony':  (0, 0, 0),
    'synchrony_entropy':(0, 0, 0),
}

datasets = [
    ('Bizzego', evidence_bizzego, 61, 'structured video'),
    ('Lerique', evidence_lerique, 31, 'PCE haptic'),
    ('Gordon', evidence_gordon, 46, 'shepherd game'),
    ('Andersen', evidence_andersen, 20, 'haunted house'),
]

print('CROSS-DATASET FEATURE EVIDENCE MATRIX')
print('=' * 70)
header = '{:>20s}'.format('Feature')
for name, _, _, _ in datasets:
    header += ' {:>10s}'.format(name)
header += ' {:>4s} {:>12s}'.format('SUM', 'verdict')
print(header)
print('-' * 70)

for feat in features:
    total = 0
    line = '{:>20s}'.format(feat)
    for _, ev, _, _ in datasets:
        score = sum(ev.get(feat, (0, 0, 0)))
        total += score
        tag = '+++' if score >= 4 else ('++' if score >= 3 else ('+' if score >= 1 else '-'))
        line += ' {:>10s}'.format(tag)
    verdict = 'ROBUST' if total >= 8 else ('PROMISING' if total >= 4 else ('MARGINAL' if total >= 1 else 'WEAK'))
    line += ' {:>4d} {:>12s}'.format(total, verdict)
    print(line)

print()
print('Score: 0=no, 1=some, 2=strong evidence per (condition × surrogate × across-stim)')
print()
print('=== RECOMMENDATIONS BY PARADIGM ===')
print('Structured/event-based (PCE, task blocks):')
print('  peak_amplitude, switching_rate (ECG), RESP peak_amplitude')
print('Continuous/free-play (shepherd game, naturalistic):')
print('  peak_amplitude, WCLC switching_rate')
print('Mixed/no-interaction (video stimulation):')
print('  peak_amplitude, switching_rate, dwell_time (needs across-stim)')
print('Universal (all 4 datasets):')
print('  peak_amplitude -- only feature with nonzero signal everywhere')
