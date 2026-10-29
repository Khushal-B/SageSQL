"""Two-stage safety guardrail: intent classifier + rule-based SQL parser."""
import sqlglot
from sqlglot import exp
from transformers import pipeline

WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
    exp.Alter, exp.Merge, exp.Command, exp.TruncateTable,
)


class IntentClassifier:
    """Fine-tuned classifier that flags questions with write intent."""

    def __init__(self, model_path="outputs/intent_classifier"):
        self.clf = pipeline("text-classification", model=model_path)

    def is_write(self, question):
        out = self.clf(question)[0]
        return out["label"] == "WRITE"


def sql_is_safe(sql):
    """Rule-based check: only a single read-only SELECT statement is allowed."""
    try:
        statements = sqlglot.parse(sql)
    except sqlglot.errors.ParseError:
        return False
    if len(statements) != 1 or statements[0] is None:
        return False
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        return False
    return not any(tree.find(n) for n in WRITE_NODES)
