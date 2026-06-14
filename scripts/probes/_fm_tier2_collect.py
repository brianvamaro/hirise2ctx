import json
from pathlib import Path
root = Path("models/fang_tier2")
rows = []
for d in sorted(root.glob("tier2_*/*/metrics.json")):
    m = json.loads(d.read_text())
    a = m["aggregate"]; pf = m["per_fold"]
    label = d.parent.parent.name.replace("tier2_", "")
    thr = pf[0].get("meaningful_threshold") if pf else None
    rows.append((label, thr, a.get("spearman_rho_mean"), a.get("meaningful_auc_mean"),
                 a.get("pr_auc_mean"), a.get("precision_at_top_5pct_mean"), a.get("rmse_log1p_mean")))
print(f"{'cell':52s} {'thr':>5s} {'rho':>7s} {'mAUC':>6s} {'prAUC':>6s} {'p@5':>6s} {'rmseL':>6s}")
for label, thr, rho, mauc, pr, p5, rl in rows:
    def f(x, d=4): return f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)
    print(f"{label:52s} {f(thr,1):>5s} {f(rho):>7s} {f(mauc,3):>6s} {f(pr,3):>6s} {f(p5,3):>6s} {f(rl,3):>6s}")
