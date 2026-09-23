"""The four rungs, one app.

Rung 1  describe   -- the dashboard, reading records out of Supabase
Rung 2  predict    -- two model families side by side, and the leakage story
Rung 3  recommend  -- versioned bands, gpt-oss-120b as the second opinion, and
                      the loop: a verdict on each recommendation, and a second
                      agent that rewrites the bands from those verdicts
Rung 4  production -- write the predictions back to a table, deploy, and count
                      how much of the loop has closed

Nothing here is dataset-specific: every number and every column name in the
captions is computed from the table that was actually loaded. The only file
that changes between datasets is ladder.toml.

Run locally:  streamlit run src/app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
APP_ROOT = Path(__file__).resolve().parents[1]

import agent as agent_mod  # noqa: E402
import data as data_mod  # noqa: E402
import handoff  # noqa: E402
import insights as insights_mod  # noqa: E402
import learner as learner_mod  # noqa: E402
import access
import rules_chat  # noqa: E402
import model as model_mod  # noqa: E402
import rules_store  # noqa: E402
from rungs import RUNGS, SUBHEADS  # noqa: E402

st.set_page_config(page_title="From data to production", page_icon="📶", layout="wide")

# Secrets become environment variables, so the same code reads a key whether it
# was pasted at the rung-3 gate, written into .streamlit/secrets.toml, or typed
# into Streamlit Cloud's secrets box.
try:
    for _name, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_name, _value)
except Exception:  # noqa: BLE001 -- no secrets file is the normal local case
    pass

ACCENT = "#2563eb"
CONTRA = "#dc2626"


@st.cache_resource(show_spinner="Training the model...")
def get_model_for(df: pd.DataFrame, _spec: model_mod.Spec,
                  estimator: str) -> model_mod.TrainedModel:
    """One trained pipeline per estimator, keyed on the estimator's name.

    Streamlit's underscore rule leaves `_spec` out of the cache key. Asking
    this function for two families with the spec alone would therefore have
    handed back the same pipeline twice, with no error and no warning. The
    plain string is what makes the two calls distinct.
    """
    path = model_mod.model_path(APP_ROOT, estimator)
    loaded = model_mod.load_model(path)
    if loaded is not None:
        loaded.origin = str(path.relative_to(APP_ROOT))
        return loaded
    return model_mod.train_model(df, replace(_spec, estimator=estimator))


def get_model(df: pd.DataFrame, spec: model_mod.Spec) -> model_mod.TrainedModel:
    """The family ladder.toml chose. Everything else on the page uses this."""
    return get_model_for(df, spec, spec.estimator)


@st.cache_data(show_spinner="Running the leakage comparison...")
def get_leakage(df: pd.DataFrame, _spec: model_mod.Spec):
    honest, leaky = model_mod.leakage_demo(df, _spec)
    return honest.as_row(), leaky.as_row()


def main() -> None:
    spec = model_mod.Spec.from_toml()
    load = data_mod.load_records()
    df = load.df

    st.title("From a table to something that runs")
    st.caption(
        f"{'Live' if load.source == 'supabase' else 'Local mode'}: {load.detail}"
    )

    tab1, tab2, tab3, tab4 = st.tabs(list(RUNGS))
    with tab1:
        rung_describe(df, spec)
    with tab2:
        rung_predict(df, spec)
    with tab3:
        rung_recommend(df, spec)
    with tab4:
        rung_production(df, spec, load)


# --------------------------------------------------------------------------
# Rung 1
# --------------------------------------------------------------------------
def rung_describe(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[0])

    positive_rate = (df[spec.target].astype(str).str.strip() == spec.positive_label).mean()
    missing = insights_mod.missing_report(df, spec.id_column, spec.target)
    numeric, _ = insights_mod.split_columns(
        df, spec.id_column, spec.target, spec.categorical_int_max_unique
    )
    st.caption(
        f"{len(df):,} records · {positive_rate:.1%} are '{spec.positive_label}' · "
        f"{len(missing)} column{'s' if len(missing) != 1 else ''} with NULLs."
    )

    st.markdown("**One: the split is uneven, and that decides the whole evening.**")
    counts = df[spec.target].value_counts().reset_index()
    counts.columns = [spec.target, "count"]
    st.plotly_chart(
        px.bar(
            counts, x=spec.target, y="count", color=spec.target,
            color_discrete_sequence=[ACCENT, CONTRA],
            title=f"{spec.target}: {positive_rate:.1%} are '{spec.positive_label}'",
        ),
        width="stretch",
    )
    st.caption(
        f"Because {1 - positive_rate:.0%} are not '{spec.positive_label}', a model "
        f"that always says no is {1 - positive_rate:.0%} accurate and worth nothing. "
        "That is rung 2's problem, and it starts here."
    )

    st.markdown(f"**Two: what moves with {spec.target}.**")
    linked = insights_mod.associations(
        df, spec.target, spec.positive_label, spec.id_column,
        spec.categorical_int_max_unique,
    )
    if linked:
        frame = pd.DataFrame(linked)
        figure = px.bar(
            frame, x="strength", y="column", color="kind", orientation="h",
            color_discrete_map={"numeric": ACCENT, "categorical": CONTRA},
            title=f"How strongly each column moves with {spec.target} (0 = not at all)",
            category_orders={"column": frame["column"].tolist()},
        )
        figure.update_yaxes(autorange="reversed")   # strongest on top
        st.plotly_chart(figure, width="stretch")
        top = linked[0]
        st.caption(
            f"{top['column']} carries the most signal ({top['strength']:.2f}); "
            f"{linked[-1]['column']} carries almost none ({linked[-1]['strength']:.2f}). "
            "Correlation for a measurement, Cramér's V for a category, so one "
            "chart can hold the whole table."
        )

    separated = insights_mod.numeric_separation(
        df, spec.target, spec.positive_label, numeric
    )[:3]
    if separated:
        st.markdown("**Three: where the two groups sit on the measurements.**")
        columns = st.columns(len(separated))
        for slot, row in zip(columns, separated):
            slot.plotly_chart(
                px.histogram(
                    df, x=row["column"], color=spec.target, barmode="overlay",
                    histnorm="percent", nbins=30, opacity=0.65,
                    color_discrete_sequence=[ACCENT, CONTRA],
                    title=row["column"],
                ),
                width="stretch",
            )
        first = separated[0]
        st.caption(
            f"Median {first['column']}: {first['median_positive']:,.0f} for "
            f"'{spec.positive_label}' against {first['median_negative']:,.0f} for the rest. "
            "Each bar is a share of its own group, so the two shapes compare "
            "even though the groups differ in size."
        )

    st.markdown("**Four: who goes with whom.**")
    ranked = insights_mod.discriminative_categoricals(
        df, spec.target, spec.positive_label, k=2
    )
    if not ranked:
        st.write("No categorical column separates the two groups. That is a finding.")
    else:
        columns = st.columns(len(ranked))
        for slot, row in zip(columns, ranked):
            rates = (
                pd.DataFrame(
                    {"value": list(row["rates"]), "rate": list(row["rates"].values())}
                )
                .sort_values("rate", ascending=False)
            )
            slot.plotly_chart(
                px.bar(
                    rates, x="value", y="rate",
                    color_discrete_sequence=[ACCENT],
                    title=f"'{spec.positive_label}' rate by {row['column']}",
                ),
                width="stretch",
            )
        first = ranked[0]
        hi = max(first["rates"], key=first["rates"].get)
        lo = min(first["rates"], key=first["rates"].get)
        st.caption(
            f"{first['column']} = {hi} is at {first['rates'][hi]:.0%}; {lo} is at "
            f"{first['rates'][lo]:.0%} -- a spread of {first['spread']:.0%}. That gap is "
            "why rung 3's bands are worth setting at all."
        )

    if len(numeric) >= 2:
        st.markdown("**Five: what is correlated with what.**")
        matrix = insights_mod.correlation_matrix(
            df, spec.target, spec.positive_label, numeric
        )
        st.plotly_chart(
            px.imshow(
                matrix, text_auto=".2f", zmin=-1, zmax=1,
                color_continuous_scale="RdBu",
                title="Pearson correlation; the target is 0/1",
            ),
            width="stretch",
        )
        off = matrix.where(~pd.DataFrame(
            np.eye(len(matrix), dtype=bool),
            index=matrix.index, columns=matrix.columns,
        ))
        pair = off.drop(columns=spec.target).drop(index=spec.target).abs().stack().idxmax()
        st.caption(
            f"{pair[0]} and {pair[1]} move together "
            f"({matrix.loc[pair[0], pair[1]]:.2f}); the model does not need both. "
            f"The last column is what each measurement does to {spec.target}."
        )

    with st.expander("What is missing, and what the blanks were hiding"):
        if not missing:
            st.write("No column has a NULL. That is rarer than it sounds -- say so out loud.")
        else:
            top = missing[0]
            explanation = insights_mod.explains_missing(df, top["column"])
            sentence = (
                f"{top['nulls']} records have no {top['column']} "
                f"({top['share']:.1%} of the table)."
            )
            if explanation:
                sentence += (
                    f" Every one of them has {explanation['column']} = "
                    f"{explanation['value']} -- the blank is a fact about those rows, "
                    "not a data error. The mistake is to coerce it to zero."
                )
            st.write(sentence)
            st.dataframe(
                df[df[top["column"]].isna()].head(25),
                width="stretch", hide_index=True,
            )
            if len(missing) > 1:
                st.caption(
                    "Other columns with NULLs: "
                    + ", ".join(f"{m['column']} ({m['nulls']})" for m in missing[1:])
                )


# --------------------------------------------------------------------------
# Rung 2
# --------------------------------------------------------------------------
def rung_predict(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[1])

    # Two model families side by side, the chosen one first. The majority-class
    # baseline is still computed (model.py, REPORT data) but not shown: on stage
    # it opened a recall discussion the talk does not have room for (23.9.2026).
    families = [spec.estimator] + [e for e in model_mod.ESTIMATORS if e != spec.estimator]
    st.dataframe(
        pd.DataFrame([get_model_for(df, spec, e).metrics.as_row() for e in families]),
        width="stretch", hide_index=True,
    )
    st.caption(
        f"Same data, same split, two kinds of model. "
        f"{model_mod.ESTIMATORS.get(spec.estimator, spec.estimator)} is the one the "
        "rest of the app uses."
    )
    with st.expander("Why the number can lie -- the leakage demo"):
        honest, leaky = get_leakage(df, spec)
        st.dataframe(pd.DataFrame([leaky, honest]), width="stretch", hide_index=True)
        st.caption(
            f"Balance the classes before splitting and the same model reports "
            f"{leaky['recall']:.0%} recall; split first and it is {honest['recall']:.0%}. "
            "One line of code apart."
        )

    st.divider()
    scoring_form(df, spec)


FORM_NUMERIC = ("tenure", "monthly_charges")
FORM_CATEGORICAL_MAX = 4


def scoring_form(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    """Score a customer who is not in the table, on both model families.

    Six fields, and which six is a question asked of the data rather than a list
    written here: the strongest categorical separators come from the same
    function rung 1 uses. The other columns are filled from the table and shown,
    because a default nobody can see is a number nobody can argue with.
    """
    numeric, categorical = model_mod.split_columns(df, spec)
    defaults = model_mod.feature_defaults(df, spec)
    drivers = [
        row["column"]
        for row in insights_mod.discriminative_categoricals(
            df, spec.target, spec.positive_label, k=FORM_CATEGORICAL_MAX
        )
        if row["column"] in categorical
    ]
    numbers = [column for column in FORM_NUMERIC if column in numeric]
    if not numbers:  # a dataset without the Telco columns: its first two numbers
        numbers = list(numeric)[:2]

    st.markdown("**Score a customer who is not in the table.**")
    record = dict(defaults)
    with st.form("score_one"):
        slots = st.columns(2)
        for position, column in enumerate(drivers + numbers):
            slot = slots[position % 2]
            label = column.replace("_", " ")
            if column in drivers:
                values = sorted(df[column].dropna().unique().tolist(), key=str)
                start = values.index(defaults[column]) if defaults[column] in values else 0
                record[column] = slot.selectbox(label, values, index=start)
            else:
                record[column] = slot.number_input(
                    label,
                    min_value=float(df[column].min()),
                    max_value=float(df[column].max()),
                    value=float(defaults[column]),
                )
        submitted = st.form_submit_button("Score this customer", type="primary")

    if not submitted:
        st.caption(
            f"{len(drivers) + len(numbers)} fields are yours; the remaining "
            f"{len(numeric) + len(categorical) - len(drivers) - len(numbers)} come "
            "from the table and are listed with the result."
        )
        return

    derived = []
    if "total_charges" in numeric and {"tenure", "monthly_charges"} <= set(record):
        record["total_charges"] = float(record["tenure"]) * float(
            record["monthly_charges"]
        )
        derived.append("total_charges")

    bands = agent_mod.Bands.load(conn=data_mod.get_connection())
    probabilities: dict[str, float] = {
        estimator: model_mod.score_record(get_model_for(df, spec, estimator), record)
        for estimator in model_mod.ESTIMATORS
    }
    # One number decides. The other family is shown small, as a comparison,
    # so nobody has to ask which of two big numbers the app acts on.
    chosen = probabilities[spec.estimator]
    st.metric(
        f"P({spec.target} = {spec.positive_label}) -- "
        f"{model_mod.ESTIMATORS.get(spec.estimator, spec.estimator)}",
        f"{chosen:.1%}",
    )
    st.caption(
        f"Rung 3 acts on this number: {bands.recommend({'probability': chosen}).action}. "
        + " · ".join(
            f"{model_mod.ESTIMATORS.get(e, e)} would say {p:.1%}"
            for e, p in probabilities.items() if e != spec.estimator
        )
        + " -- a different model, a different number; only the chosen one decides."
    )

    typed = {column: record[column] for column in drivers + numbers}
    handoff.stash_scored(st.session_state, record, probabilities, typed)
    st.session_state.pop("table_record", None)
    st.caption("This customer is now the one rung 3 opens on.")

    filled = {
        column: value
        for column, value in record.items()
        if column not in drivers + numbers
    }
    with st.expander(f"{len(filled)} fields filled in from the table"):
        if derived:
            st.caption(
                "total_charges is derived as tenure x monthly charges, not taken "
                "from the table: a median total against a one-month tenure would "
                "be a contradiction to feed the model."
            )
        st.dataframe(
            pd.DataFrame(
                {"column": list(filled), "value": [filled[c] for c in filled]}
            ),
            width="stretch",
            hide_index=True,
        )


# --------------------------------------------------------------------------
# Rung 3
# --------------------------------------------------------------------------
def rung_recommend(df: pd.DataFrame, spec: model_mod.Spec) -> None:
    st.subheader(SUBHEADS[2])

    conn = data_mod.get_connection()
    trained = get_model(df, spec)
    ranked = insights_mod.discriminative_categoricals(
        df, spec.target, spec.positive_label, k=2
    )
    extra = [row["column"] for row in ranked]
    scored = model_mod.score_records(trained, df, spec, extra)
    rules_set = rules_store.load_active(conn)
    bands = agent_mod.Bands.from_rules(rules_set)
    feedback = data_mod.read_feedback(conn)
    st.caption(f"A probability is not a decision. Rules v{rules_set.version} turn it into one.")

    top = scored.head(25)
    labels = [
        f"{row[spec.id_column]} · risk {row['probability']:.0%}"
        for _, row in top.iterrows()
    ]
    default = handoff.default_record(st.session_state, spec.estimator, spec.id_column)
    override = st.session_state.get("table_record")

    if override is not None:
        record = override
        who = f"{record[spec.id_column]}, from the table"
    elif default is not None:
        record = default
        typed = st.session_state[handoff.KEY]["typed"]
        who = " · ".join(
            value if isinstance(value, str) and value not in ("Yes", "No")
            else f"{field.replace('_', ' ')} {value:g}" if not isinstance(value, str)
            else f"{field.replace('_', ' ')} {value}"
            for field, value in typed.items()
        )
    else:
        st.caption("Nothing scored on rung 2 yet -- score a customer there and it opens here first.")
        choice = st.selectbox(
            "Pick a customer from the table", range(len(labels)),
            format_func=lambda i: labels[i],
        )
        record = top.iloc[choice].to_dict()
        who = f"{record[spec.id_column]}, from the table"

    rules = bands.recommend(record)
    st.markdown(f"**{who}**")
    st.success(rules.action)
    st.caption(rules.reason)

    st.divider()
    rules_chat_panel(conn, rules_set, feedback, record, str(record[spec.id_column]))
    feedback_form(conn, record, rules, rules_set, spec, df)

    if default is not None:
        with st.expander("Try another customer from the table"):
            st.caption(
                "Walking real customers one at a time is how you find out whether "
                "the bands are right. It is the way to formulate the rules the agent "
                "will act on, not only to read them."
            )
            choice = st.selectbox(
                "Pick a customer", range(len(labels)),
                format_func=lambda i: labels[i], key="table_pick",
            )
            c1, c2 = st.columns(2)
            if c1.button("Use this customer", key="use_table"):
                st.session_state["table_record"] = top.iloc[choice].to_dict()
                st.rerun()
            if override is not None and c2.button(
                "Back to the customer from rung 2", key="back_rung2"
            ):
                st.session_state.pop("table_record", None)
                st.rerun()


# --------------------------------------------------------------------------
# The loop -- a verdict on the recommendation, and the agent that learns from it
# --------------------------------------------------------------------------
def outcome_labels(spec, df: pd.DataFrame | None = None) -> dict[str, str]:
    """The three answers to "what actually happened", in the target's own words.

    The stored codes stay `stayed` / `left` (the event did not / did happen) so
    the learner and the database never change; only what the form shows does.
    """
    negative = f"not {spec.positive_label}"
    if df is not None:
        others = [
            str(v) for v in df[spec.target].dropna().astype(str).str.strip().unique()
            if str(v) != spec.positive_label
        ]
        negative = others[0] if len(others) == 1 else negative
    return {
        "unknown": "don't know yet",
        "stayed": f"{spec.target} = {negative}",
        "left": f"{spec.target} = {spec.positive_label}",
    }


def rules_chat_panel(conn, rules_set, feedback: list[dict], record: dict,
                     record_id: str) -> None:
    """Approach A: the agent talks; a button applies.

    The conversation is keyed to the rules version. When the learner or this
    panel publishes a new version the page reruns, the key changes, and the
    conversation starts over on the new rules -- so the console can never be
    arguing about bands that are no longer live.
    """
    st.markdown("##### Talk to the rules")
    key = rules_chat.chat_key(rules_set, record_id)
    if st.session_state.get("rules_chat_key") != key:
        previous = st.session_state.get("rules_chat_key", "")
        restarted = bool(previous) and not previous.startswith(f"rules_chat_v{rules_set.version}_")
        st.session_state["rules_chat_key"] = key
        st.session_state["rules_chat"] = [
            {"role": "assistant", "content": rules_chat.opening_message(rules_set, record)}
        ]
        st.session_state.pop("chat_proposal", None)
        if restarted:
            st.caption(f"The rules moved to v{rules_set.version}; the conversation starts over on them.")

    for turn in st.session_state["rules_chat"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if not rules_chat.llm_available():
        st.info(
            "GROQ_API_KEY is not set, so the console can state the rules but not "
            "discuss them. That is the honest state of the demo, not a failure."
        )
        return

    prompt = st.chat_input("Ask about the rules, or say what should change")
    if prompt:
        allowed, reason = access.chat_allowed()
        if not allowed:
            st.info(reason)
            return
        access.note_chat_call()
        st.session_state["rules_chat"].append(
            {"role": "user", "content": access.clean_text(prompt, 500) or ""}
        )
        digest_rows = (
            [d.as_row() for d in learner_mod.digest(rules_set, feedback).values()]
            if feedback else []
        )
        system = rules_chat.system_prompt(rules_set, digest_rows, record)
        with st.spinner("Thinking..."):
            turn = rules_chat.reply(st.session_state["rules_chat"], system, current=rules_set)
        if turn is None or turn.error:
            reason = turn.error if turn is not None else "GROQ_API_KEY is not set"
            st.session_state["rules_chat"].append(
                {"role": "assistant",
                 "content": f"_The model could not answer. {reason}. The rules stand._"}
            )
        else:
            st.session_state["rules_chat"].append({"role": "assistant", "content": turn.text})
            if turn.bands is not None:
                st.session_state["chat_proposal"] = {
                    "bands": turn.bands, "rationale": turn.rationale,
                    "version": rules_set.version,
                }
        st.rerun()

    proposal = st.session_state.get("chat_proposal")
    if not proposal or proposal["version"] != rules_set.version:
        return
    changes = learner_mod._diff(rules_set, proposal["bands"])
    st.markdown(f"**The console proposes v{rules_set.version + 1}**")
    if proposal["rationale"]:
        st.write(proposal["rationale"])
    if not changes:
        st.info("Identical to the current bands -- nothing to apply.")
        return
    c1, c2 = st.columns(2)
    c1.markdown(f"v{rules_set.version} (now)")
    c1.dataframe(pd.DataFrame(rules_set.as_rows()), width="stretch", hide_index=True)
    c2.markdown(f"v{rules_set.version + 1} (proposed)")
    c2.dataframe(
        pd.DataFrame([b.__dict__ for b in proposal["bands"]]),
        width="stretch", hide_index=True,
    )
    for line in changes:
        st.markdown(f"- {line}")
    if conn is None:
        st.warning("No Supabase connection -- the proposal can be seen but not applied.")
        return
    if not access.is_editor():
        st.button(f"Apply as v{rules_set.version + 1}", disabled=True, key="chat_apply_locked")
        st.caption(access.LOCKED_NOTE)
        return
    if st.button(f"Apply as v{rules_set.version + 1}", type="primary", key="chat_apply"):
        published = rules_store.publish(
            conn, proposal["bands"], "chat:groq", proposal["rationale"],
            {"chat": st.session_state["rules_chat"][-6:]},
        )
        st.session_state.pop("chat_proposal", None)
        st.success(
            f"Rules v{published.version} is active. The recommendation above now "
            "comes from it."
        )
        st.rerun()


def feedback_form(conn, record: dict, rules, rules_set, spec,
                  df: pd.DataFrame | None = None) -> None:
    """One verdict per recommendation. This is the input the whole loop runs on."""
    labels = outcome_labels(spec, df)
    st.markdown("##### Was this the right call? Tell the system.")
    record_id = str(record[spec.id_column])
    logged = data_mod.latest_prediction_for(conn, record_id)
    with st.form(f"feedback_{record_id}", clear_on_submit=True):
        c1, c2 = st.columns(2)
        verdict = c1.radio("The recommendation was", ["right", "wrong"], horizontal=True)
        outcome = c2.radio(
            "What actually happened", list(labels), horizontal=True,
            format_func=labels.get,
        )
        better = st.text_input(
            "What should have been done instead? (one line; empty if the call was right)"
        )
        with st.expander("Why? (optional)"):
            note = st.text_input("A sentence for the learner to read")
        sent = st.form_submit_button("Send feedback", type="primary")
    if sent:
        row = {
            "prediction_id": (logged or {}).get("id"),
            "record_id": record_id,
            "probability": round(float(record.get("probability") or 0.0), 4),
            "recommended_action": rules.action,
            "verdict": verdict,
            "actual_outcome": outcome,
            "better_action": access.clean_text(better),
            "note": access.clean_text(note),
            "rules_version": rules_set.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        ok, message = data_mod.write_feedback(conn, row)
        (st.success if ok else st.error)(message)
        if ok:
            _band_progress(conn, rules_set, row)
        if ok and logged:
            st.caption(
                f"Attached to prediction #{logged['id']} from "
                f"{str(logged.get('created_at', ''))[:16]} -- its outcome column is filled now."
            )
    elif logged is None:
        st.caption(
            "No logged decision for this record yet -- the feedback is still saved, "
            "it just has no prediction row to attach an outcome to. Rung 4 writes those."
        )


def _band_progress(conn, rules_set, row: dict) -> None:
    """After a send: how far this band is from the learner's threshold, and what
    the verdicts so far would make it do. The '3' used to live only in Python."""
    t = learner_mod.thresholds()
    band = rules_set.band_for(float(row.get("probability") or 0.0)).name
    digested = learner_mod.digest(rules_set, data_mod.read_feedback(conn))
    d = digested.get(band)
    if d is None:
        return
    if d.total < t.min_verdicts:
        st.info(
            f"Verdicts on the '{band}' band: {d.total} of {t.min_verdicts} the learner "
            f"needs before it may change it ({d.wrong} say wrong)."
        )
    else:
        st.info(
            f"Verdicts on the '{band}' band: {d.total} ({d.wrong} say wrong) -- enough "
            "for the learner. Rung 4: 'Let the learner revise the rules'."
        )
    if row.get("verdict") == "wrong" and not row.get("better_action"):
        st.caption(
            "No better action was named, so the learner cannot change what this band "
            "recommends -- it can only move a floor. Say what should have been done."
        )


def learner_panel(conn, rules_set, feedback: list[dict]) -> None:
    """Agent 2: reads the verdicts, proposes a revision, publishes it on request."""
    st.caption(learner_mod.thresholds().rule_text())
    if not feedback:
        st.caption("No verdicts yet. Send one on rung 3 and the learner has something to read.")
        return

    prefer_llm = learner_mod.llm_available()
    writer = ("the rule-based learner, with gpt-oss-120b as a second opinion when "
              "the rules find nothing") if prefer_llm else "the rule-based learner"
    if st.button("Let the learner revise the rules", type="primary"):
        with st.spinner("Reading the verdicts..."):
            proposal = learner_mod.propose(rules_set, feedback, prefer_llm=prefer_llm)
        st.session_state["proposal"] = proposal
    st.caption(f"{len(feedback)} verdicts on rules v{rules_set.version}; {writer}.")
    with st.expander("What the learner read, folded per band"):
        digested = learner_mod.digest(rules_set, feedback)
        st.dataframe(
            pd.DataFrame([d.as_row() for d in digested.values()]),
            width="stretch", hide_index=True,
        )

    proposal = st.session_state.get("proposal")
    if proposal is None or proposal.before.version != rules_set.version:
        return

    st.write(proposal.rationale)
    if not proposal.changed:
        st.info("The evidence does not move any band yet. That is a finding, not a failure.")
        return
    for line in proposal.changes:
        st.markdown(f"- {line}")

    if conn is None:
        st.warning("No Supabase connection -- the revision can be seen but not published.")
        return
    if not access.is_editor():
        st.button(f"Publish v{proposal.before.version + 1} -- the recommender uses it from now on",
                  disabled=True, key="publish_locked")
        st.caption(access.LOCKED_NOTE)
        return
    if st.button(f"Publish v{proposal.before.version + 1} -- the recommender uses it from now on",
                 type="secondary"):
        published = rules_store.publish(
            conn, proposal.bands, proposal.source, proposal.rationale, proposal.evidence
        )
        st.session_state.pop("proposal", None)
        st.success(
            f"Rules v{published.version} is active. Back on rung 3, the "
            "recommendation now comes from the revised bands."
        )
        st.rerun()


# --------------------------------------------------------------------------
# Rung 4
# --------------------------------------------------------------------------
def rung_production(df: pd.DataFrame, spec: model_mod.Spec,
                    load: data_mod.LoadResult) -> None:
    st.subheader(SUBHEADS[3])
    st.caption(
        "A job scores the table and writes the riskiest rows down (the audit "
        "trail at the bottom); people send verdicts on rung 3; a second agent "
        "rewrites the rules from them."
    )

    conn = data_mod.get_connection()
    trained = get_model(df, spec)
    scored = model_mod.score_records(trained, df, spec, [])
    rules_set = rules_store.load_active(conn)
    source = "Supabase" if load.source == "supabase" else "the local CSV"

    st.caption(f"Model {trained.version} from {getattr(trained, 'origin', 'this process')} · "
               f"rules v{rules_set.version} · {len(scored):,} customers scored from {source}.")

    st.markdown("**1 · Learn from what came back**")
    counts = data_mod.loop_counts(conn)
    st.caption(f"{counts['logged']:,} decisions logged · "
               f"{counts['verdicts']:,} verdicts from people on rung 3.")
    feedback = data_mod.read_feedback(conn)
    learner_panel(conn, rules_set, feedback)

    st.divider()
    st.markdown("**2 · Who changed the rules, and when**")
    versions = rules_store.history(conn, limit=5)
    if not versions:
        st.caption("No Supabase connection, so no history: the rules are ladder.toml's v1.")
    else:
        for row in versions:
            when = str(row.get("created_at", ""))[:16].replace("T", " ")
            st.markdown(f"- v{row['version']} · {row.get('source', '')} · {when} -- "
                        f"{row.get('rationale', '') or 'initial bands from ladder.toml'}")
        if len(versions) >= 2 and versions[0].get("bands") is not None:
            newest = rules_store.from_row(versions[0])
            previous = rules_store.from_row(versions[1])
            for line in learner_mod._diff(previous, newest.bands):
                st.caption(f"v{previous.version} -> v{newest.version}: {line}")

    with st.expander("Audit trail -- the predictions table"):
        recent = data_mod.read_predictions(conn)
        if recent.empty:
            st.info("Nothing written yet -- the job is: python ladder.py app")
        else:
            st.dataframe(recent, width="stretch", hide_index=True)

    with st.expander("Latest verdicts from people"):
        verdicts = data_mod.read_feedback(conn, limit=10)
        if not verdicts:
            st.info("No feedback yet. Rung 3 collects it under every recommendation.")
        else:
            frame = pd.DataFrame(verdicts[::-1])
            columns = ("created_at", "record_id", "recommended_action", "verdict",
                       "actual_outcome", "better_action", "rules_version")
            st.dataframe(
                frame[[c for c in columns if c in frame.columns]],
                width="stretch", hide_index=True,
            )


if __name__ == "__main__":
    main()
