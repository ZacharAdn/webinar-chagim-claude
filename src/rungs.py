"""The four rungs' labels, in one place with no Streamlit import.

app.py renders them; app_step.py asserts on them. Neither should have to import
the other to agree on four strings.
"""

RUNGS = ("1 · Describe", "2 · Predict", "3 · Recommend", "4 · Production")
SUBHEADS = (
    "What is actually in the table",
    "A number, and whether you can believe it",
    "From a number to something you can act on",
    "The part that makes it real",
)
