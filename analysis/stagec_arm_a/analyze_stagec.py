import csv, json, statistics, sys, os

def load_metrics(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['step']), float(r['train_acc']), float(r['val_acc'])))
    return rows

def load_steps(path):
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            rows.append((int(r['step']), float(r['loss']), float(r['grad_norm']), float(r['update_to_weight_ratio'])))
    return rows

def episodes(rows, x, cross_step, key=2):
    running_max = None; eps = []; cur = None
    for step, tr, va in rows:
        if step < cross_step: continue
        a = tr if key==1 else va
        if running_max is None or a > running_max: running_max = a
        drop = running_max - a
        if drop >= x:
            if cur is None: cur = {'onset': step, 'depth': drop, 'last': step}
            else:
                cur['last'] = step
                if drop > cur['depth']: cur['depth'] = drop
        else:
            if cur is not None: eps.append(cur); cur = None
    if cur is not None: cur['open'] = True; eps.append(cur)
    return eps

def first_crossing(rows, key=2, thresh=99.0):
    for step, tr, va in rows:
        if (tr if key==1 else va) >= thresh: return step

def med(xs): return statistics.median(xs) if xs else None

def analyze(name, mpath, spath):
    print(f"########## {name} ##########")
    rows = load_metrics(mpath)
    cs = first_crossing(rows)
    print(f"first >=99% val crossing: step {cs}")
    for x in (2.0, 5.0, 10.0):
        eps = episodes(rows, x, cs)
        opens = [e for e in eps if e.get('open')]
        print(f"  {x}pt: n={len(eps)} median_depth={med([e['depth'] for e in eps]):.4g} open={len(opens)}" + (f" (open onset step {opens[0]['onset']}, depth {opens[0]['depth']:.1f}, last eval {opens[0]['last']})" if opens else ""))
    eps5 = episodes(rows, 5.0, cs)
    deep = max(eps5, key=lambda e: e['depth'])
    print(f"  deepest 5pt episode: {deep['depth']:.1f}pt at step {deep['onset']}..{deep['last']}")
    # float32 saturation probe: loss floor
    steps = load_steps(spath)
    losses = [l for _,l,_,_ in steps]
    below = sum(1 for l in losses if l < 1e-6)
    print(f"  float32 CE saturation probe: loss floor min={min(losses):.4g}, steps below 1e-6: {below}")
    # precede-vs-accompany probe on update_to_weight_ratio
    uw = {s: u for s,_,_,u in steps}
    all_uw = [u for _,_,_,u in steps]
    base = med(all_uw)
    pre = []
    for e in eps5:
        w = [uw[s] for s in range(e['onset']-50, e['onset']) if s in uw]
        if w: pre.extend(w)
    print(f"  update_to_weight_ratio: overall median {base:.3g}; 50-steps-pre-onset median {med(pre):.3g} (n={len(pre)}) - DESCRIPTIVE")
    # grad_norm spike-precedes-dip (original telemetry method)
    gns = sorted(g for _,_,g,_ in steps)
    p95 = gns[int(0.95*(len(gns)-1))]
    hits = 0
    for e in eps5:
        win = [g for s,_,g,_ in steps if e['onset']-50 <= s < e['onset']]
        if win and max(win) > p95: hits += 1
    print(f"  grad_norm p95={p95:.4g}; 5pt-episode onsets with >p95 grad_norm in prior 50 steps: {hits}/{len(eps5)} - DESCRIPTIVE count, no pre-specified null")

# Usage: python3 analyze_stagec_rebuilt.py <control_dir> <stagec_export_dir>
#   control_dir: the original seed1_cpu_control_export (metrics.csv, diagnostics_steps.csv)
#   stagec_export_dir: the Stage C bundle root (eps1e-4_arm/, tail_replay/)

CONTROL = sys.argv[1] if len(sys.argv) > 1 else "seed1_cpu_control_export"
BUNDLE = sys.argv[2] if len(sys.argv) > 2 else "stagec_export"
analyze("CONTROL (seed1 eps 1e-8, byte-identical replay)", os.path.join(CONTROL, "metrics.csv"), os.path.join(CONTROL, "diagnostics_steps.csv"))
print()
analyze("ARM A (seed1 eps 1e-4)", os.path.join(BUNDLE, "eps1e-4_arm", "metrics.csv"), os.path.join(BUNDLE, "eps1e-4_arm", "diagnostics_steps.csv"))

# dense tail: extend-control view - any new episodes in 98950..100000 tail?
tail = []
with open(os.path.join(BUNDLE, "tail_replay", "tail_evals.csv"), newline='') as f:
    for r in csv.DictReader(f):
        tail.append((int(r['step']), float(r['train_acc']), float(r['val_acc'])))
print(f"\nDENSE TAIL (98950..100000, {len(tail)} rows): val_acc first={tail[0][2]:.2f} last={tail[-1][2]:.2f} min={min(t[2] for t in tail):.2f} max={max(t[2] for t in tail):.2f}")
print(f"  final eval step {tail[-1][0]}: train_acc={tail[-1][1]:.2f} val_acc={tail[-1][2]:.2f}")
