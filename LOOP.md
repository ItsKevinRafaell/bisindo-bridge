# LOOP.md — BISINDO Bridge Loop Plan (Journal Submission)

> **Goal**: Ship "BISINDO Bridge v1.0-journal" — a research-prototype paper showing CNN-based sign language recognition with reproducible results.
> **Scope**: CNN-only (per user 2026-07-06). ML models (RF/SVM) out of scope.
> **Format**: research prototype / proof-of-concept, NOT production-grade.

---

## The Loop (meta)

```
┌─────────────────────────────────────────────────┐
│  SPEC  →  BUILD  →  VERIFY  →  LEARN            │
│    ↑                                          │
│    └──────────── drift detected? ──────────────┘
└─────────────────────────────────────────────────┘
```

Each sub-loop:
1. **SPEC**: what outcome, what files, what verification
2. **BUILD**: execute via skill
3. **VERIFY**: run check, capture evidence
4. **LEARN**: update STATE.md, archive artifacts

---

## Loop 0 — Adopt ritual scaffolding  ·  STATUS: 🟡 in progress

- [x] Audit (read-only) — done, plan saved
- [x] Create STATE.md — done
- [x] Create LOOP.md — done
- [ ] Create `.claude/skills/` — Loop 0b
- [ ] Create `.claude/budget.md` — Loop 0c
- [ ] Drift check: STATE.md vs LOOP.md aligned

**VERIFY**: `ls STATE.md LOOP.md .claude/skills/ .claude/budget.md` → all exist

---

## Loop 1 — Reproduce CNN champion  ·  STATUS: ⬜

**SPEC**:
- Re-train CNN di env clean (lokal, GPU/CPU)
- Verify accuracy ≥ 0.985 (match published 0.991)
- Save metrics + learning curve
- Output: `models/dl/cnn_repro_metrics.json` + `models/dl/cnn_repro_curve.png`

**BUILD**: invoke skill `cnn-train`

**VERIFY**:
- `cat models/dl/cnn_repro_metrics.json` → accuracy field ≥ 0.985
- `ls models/dl/cnn_repro_curve.png` → exists
- Run eval/compare.py on the repro model → results reproducible

**LEARN**: append metrics to STATE.md, mark loop done in this file

---

## Loop 2 — Eval upgrade  ·  STATUS: ⬜

**SPEC**:
- Add normalized confusion matrix → `models/dl/cnn_confusion_matrix.png`
- Add per-class F1 bar chart → `models/dl/cnn_per_class.png`
- Add `classification_report.json` per model
- Generate `models/comparison_report.md` (CNN-only)

**BUILD**: invoke skill `cnn-eval`

**VERIFY**:
- All 3 artifacts exist
- `cat models/comparison_report.md` shows accuracy, F1, precision, recall, per-class table
- Confusion matrix visually inspected (not just blank/diagonal)

**LEARN**: append report path + key numbers to STATE.md

---

## Loop 3 — Paper docs  ·  STATUS: ⬜

**SPEC**:
- `docs/RESULTS.md` — champion model selection rationale + comparison table
- `docs/DATASET.md` — data card (rows, classes, contributors, license, intended use)
- `docs/LIMITATIONS.md` — honest constraints (CNN-only, app demo-quality, etc)
- Update `README.md` — reframe as "research prototype", link to RESULTS/DATASET/LIMITATIONS

**BUILD**: invoke skill `write-paper-docs`

**VERIFY**:
- All 4 docs exist
- Each has required sections (see spec)
- README has framing line + 3 doc links

**LEARN**: update STATE.md docs section

---

## Loop 4 — App hardening (meeting/app.py)  ·  STATUS: ⬜

**SPEC**:
- `meeting/app.py:33` SECRET_KEY → env var (`BISINDO_SECRET_KEY` fallback)
- `meeting/app.py:36` CORS → restrict to env var (`BISINDO_CORS_ORIGINS`, default `*`)
- `meeting/app.py:774` → remove `allow_unsafe_werkzeug=True` (use eventlet/gunicorn in real deploy)
- `meeting/app.py:294-296` → dedupe duplicate `if len(hand1) != 63` check
- `.gitignore` → add `meeting/node_modules/`, `meeting/cloudflared.deb`, `meeting/sfu/node_modules/`
- Decision: keep `meeting/models` symlink (works on VPS too)

**BUILD**: invoke skill `meeting-fix`

**VERIFY**:
- `grep -n "bisindo-meeting-secret-2026" meeting/app.py` → no match
- `grep -n "allow_unsafe_werkzeug" meeting/app.py` → no match
- `python -c "import ast; ast.parse(open('meeting/app.py').read())"` → no syntax error
- `cat .gitignore | grep cloudflared.deb` → matches

**LEARN**: bump restart counter / tag code version

---

## Loop 5 — VPS fixes  ·  STATUS: ⬜

**SPEC**:
- Diagnose CSV not loading: check actual file path on VPS vs what app.py expects
- Install libGLESv2 (or remove MediaPipe dep from classifier init path) so classifier loads
- Kill 1 of 2 cloudflared processes (keep systemd one, kill stray pid 316)
- Install node + npm on VPS, start SFU as separate systemd unit
- Restart bisindo-server.service, verify classifier=true + samples>0

**BUILD**: invoke skill `vps-fix`

**VERIFY**:
- `ssh root@109.176.17.11 "curl -s http://localhost:4500/api/health"` → `"classifier":true, "samples":>0`
- `ssh root@109.176.17.11 "ps aux | grep cloudflared | grep -v grep | wc -l"` → 1
- `ssh root@109.176.17.11 "curl -s http://localhost:4501/ -o /dev/null -w '%{http_code}'"` → 200 or 426 (WS upgrade)

**LEARN**: update STATE.md VPS section

---

## Loop 6 — Final smoke test  ·  STATUS: ⬜

**SPEC**:
- Run eval/compare.py end-to-end → verify report generated
- ssh VPS → verify health endpoint
- Open tunnel URL in headless browser → verify HTML loads (200)
- Tag repo: `v1.0-journal`

**BUILD**: invoke skill `smoke-test`

**VERIFY**: all green checks above

**LEARN**: archive session summary

---

## Drift Detection (STATE.md ↔ LOOP.md)

Every time STATE.md updates, check:
- [ ] Each loop marked done in STATE.md also marked done here
- [ ] No new blockers in STATE.md that aren't reflected here
- [ ] No completed loops here that aren't marked done in STATE.md

If drift detected → fix one to match the other (canonical = STATE.md, since it's the live snapshot).

---

## Skills Index (`.claude/skills/`)

| Skill | Purpose |
|---|---|
| `cnn-train` | Train CNN, save metrics + learning curve |
| `cnn-eval` | Generate confusion matrix, per-class report, comparison markdown |
| `write-paper-docs` | Generate RESULTS/DATASET/LIMITATIONS |
| `meeting-fix` | Apply security + code hardening to meeting/app.py |
| `vps-fix` | Diagnose + fix VPS deployment issues |
| `smoke-test` | End-to-end verification |

---

## Out of Scope (deferred)

- Production hardening (gunicorn, nginx, named tunnel, SSH key auth, ufw/fail2ban)
- K-fold cross-validation
- Tests + CI
- Real-time gesture tracking improvements beyond what meeting.js already does
- Mobile responsive PWA
- i18n beyond Indonesian
- ML models (RF/SVM) — CNN-only mode