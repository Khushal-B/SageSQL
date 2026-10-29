"""Spider-style evaluation on held-out databases: Execution Accuracy and Exact Match."""
import json
import sqlite3

import sqlglot

from sqlsage.pipeline import SQLSage


def normalize(sql):
    return sqlglot.parse_one(sql).sql(normalize=True)


def run(db_path, sql):
    with sqlite3.connect(db_path) as conn:
        return sorted(conn.execute(sql).fetchall(), key=str)


# data/test.json: [{"db_path": "...", "schema": {...}, "question": "...", "sql": "..."}, ...]
tests = json.load(open("data/test.json"))
sage = SQLSage()
ex = em = 0

for t in tests:
    out = sage.ask(t["question"], t["schema"])
    if out["blocked"]:
        continue
    pred, gold = out["sql"], t["sql"]
    em += normalize(pred) == normalize(gold)
    try:
        ex += run(t["db_path"], pred) == run(t["db_path"], gold)
    except sqlite3.Error:
        pass

print(f"Execution Accuracy: {ex / len(tests):.1%}")
print(f"Exact Match: {em / len(tests):.1%}")
