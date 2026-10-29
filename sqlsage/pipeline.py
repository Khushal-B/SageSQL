"""End-to-end pipeline: guardrail -> schema linking -> generation -> guardrail."""
from sqlsage.generator import SQLGenerator
from sqlsage.guardrails import IntentClassifier, sql_is_safe
from sqlsage.schema_linker import link_schema, schema_to_prompt


class SQLSage:
    def __init__(self):
        self.intent = IntentClassifier()
        self.generator = SQLGenerator()

    def ask(self, question, schema):
        if self.intent.is_write(question):
            return {"blocked": True, "reason": "write intent detected"}
        linked = link_schema(question, schema)
        sql = self.generator.generate(schema_to_prompt(linked), question)
        if not sql_is_safe(sql):
            return {"blocked": True, "reason": "unsafe SQL", "sql": sql}
        return {"blocked": False, "sql": sql}


if __name__ == "__main__":
    demo_schema = {
        "singer": ["singer_id", "name", "country", "age"],
        "concert": ["concert_id", "singer_id", "year", "venue"],
        "stadium": ["stadium_id", "name", "capacity"],
    }
    sage = SQLSage()
    print(sage.ask("How many singers are from France?", demo_schema))
