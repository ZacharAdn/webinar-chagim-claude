# webinar-chagim-claude — ladder report, 2026-09-22 23:49

| Rung | Status | Measured |
|---|---|---|
| 1 prepare | PASS | 7,043 rows · 21 columns · 11 NULLs in total_charges · 26.5% positive |
| 2 model | PASS | baseline 73.5% acc / 0% recall · model 79.3% / 52.1% / AUC 0.84 · leak 93.7% vs 48.4% · saved models/logreg.joblib (logreg-e8ac3263) |
| 3 app | PASS | 4 rungs · 0 exceptions · 25 rows to data/predictions.csv |
| 4 supabase | SKIPPED | --local-only |
| 5 github | SKIPPED | --local-only |
| 6 deploy | SKIPPED | --local-only |
| 7 verify | SKIPPED | --local-only |
| 8 feedback | SKIPPED | --local-only |

```
Rerun one step:   python ladder.py model
Rerun from step:  python ladder.py all --from supabase
Open the app:     streamlit run src/app.py
```
