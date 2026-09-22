# Plan — webinar-chagim-claude

## What this is

The four-rung ladder on `customers.csv`: describe what is in the table, predict
`churn` = `Yes`, turn the probability into an action, and put the
result where someone who was not in the room can read it.

## Why it is built this way

The code is a frozen template, not written per dataset. Every dataset-specific
fact lives in `ladder.toml`, so two projects built by this skill differ in that
one file and nowhere else. That is what makes a run reviewable in a minute:
same file names, same step order, same report, different numbers.

## The four rungs

1. **Describe.** What is in the table, what is missing and whether one column
   explains the blanks, and which categorical columns separate the two classes.
2. **Predict.** A majority-class baseline first, then the model, then the same
   model class trained on a leaky split — because the gap between the leaky and
   the honest recall is the lesson, not the model.
3. **Recommend.** Probability bands from `ladder.toml` map a number to an
   action. The rules layer is the baseline of a recommendation exactly as the
   dummy is the baseline of a prediction; Claude is an optional second opinion.
4. **Production.** Score, write rows to the `predictions` table, read them back.
   The first production version of a model is a job that writes to a table.

## Sequencing

Eight steps, fixed order, each idempotent and each recording one JSON result:
prepare, model, app, supabase, github, deploy, verify, report. Steps 1-3 run
offline. Step 6 needs a browser because Streamlit Community Cloud has no API for
creating an app. Step 7 proves the result from outside, over HTTPS, with the
publishable key.

## Risks

- **Leakage.** Oversampling before the split inflates recall. The honest and the
  leaky number are both recorded so the difference is visible, not argued.
- **RLS returning zero rows silently.** Row level security with no policy returns
  an empty table and no error. The generated schema always writes the policy.
- **The 1,000-row ceiling.** PostgREST truncates silently; `fetch_table()` pages.
- **A public app.** The free Community Cloud tier is public. Nothing from a real
  client goes into this repo.
