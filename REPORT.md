# webinar-chagim-claude — ladder report, 2026-09-23 11:59

| Rung | Status | Measured |
|---|---|---|
| 1 prepare | PASS | 7,043 rows · 21 columns · 11 NULLs in total_charges · 26.5% positive |
| 2 model | PASS | logreg 79.3% / 52.1% / AUC 0.84 · tree 78.2% / 59.1% / AUC 0.82 · leak 93.7% vs 48.4% · saved models/logreg.joblib (logreg-e8ac3263) |
| 3 app | PASS | 4 rungs · 0 exceptions · 25 rows to Supabase |
| 4 supabase | PASS | ref pffmgalxllheoddeknsh · 7,043 rows · 11 NULLs kept |
| 5 github | PASS | ZacharAdn/webinar-chagim-claude (public) |
| 6 deploy | PASS | https://webinar-chagim-claude-jnlunmnxbafl5gmr8as3up.streamlit.app |
| 7 verify | PASS | live read 7,043 · write id 26 at 07:32:21 · read back OK · app 200 |
| 8 feedback | PASS | 26 decisions logged · 0 outcomes yet — the column is open and waiting |

```
Rerun one step:   python ladder.py model
Rerun from step:  python ladder.py all --from supabase
Open the app:     streamlit run src/app.py
```
