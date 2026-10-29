"""Embedding-based schema linking: keep only the tables relevant to the question."""
from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("all-MiniLM-L6-v2")


def table_text(name, columns):
    return f"{name}: " + ", ".join(columns)


def link_schema(question, schema, top_k=3):
    """schema is a dict of {table_name: [column, ...]}. Returns the top_k closest tables."""
    names = list(schema)
    docs = [table_text(n, schema[n]) for n in names]
    doc_emb = _model.encode(docs, convert_to_tensor=True)
    q_emb = _model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, doc_emb)[0]
    top = scores.topk(min(top_k, len(names))).indices.tolist()
    return {names[i]: schema[names[i]] for i in top}


def schema_to_prompt(schema):
    return "\n".join(f"CREATE TABLE {t} ({', '.join(c)});" for t, c in schema.items())
