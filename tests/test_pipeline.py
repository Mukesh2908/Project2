import json

from dqa import llm
from dqa.cli import main
from dqa.pipeline import run_batch
from dqa.profiler import profile_dataframe
from dqa.report import render_html, render_markdown, write_reports
from dqa.rules import load_rules
from dqa.store import Store


def test_end_to_end(batches, tmp_path):
    store = Store(tmp_path / ".dqa")
    first = run_batch(batches[0], "orders", store)
    assert first.baseline_run_id is None and first.new_rules > 0 and first.verdict != "FAIL"
    assert store.rules_path("orders").exists()

    for p in batches[1:-1]:
        r = run_batch(p, "orders", store)
        assert r.counts["error"] == 0 and r.counts["warning"] == 0, [f.title for f in r.findings]

    bad = run_batch(batches[-1], "orders", store)
    assert bad.history_size == 5
    assert bad.counts["error"] >= 1  # the type change is an error even before rules are approved
    cats = {f.category for f in bad.findings}
    assert {"rule", "drift", "anomaly"} <= cats
    assert bad.new_rules > 0  # rules proposed for the new columns

    assert len(store.runs("orders")) == 6
    paths = write_reports(bad, tmp_path / "reports")
    assert json.loads(paths["json"].read_text())["verdict"] == "FAIL"
    assert "Findings" in render_markdown(bad) and "<table>" in render_html(bad)


def test_cli_run_and_approve(batches, tmp_path, capsys):
    home = str(tmp_path / ".dqa")
    assert main(["run", str(batches[0]), "--dataset", "orders", "--home", home]) == 0
    assert main(["approve", "orders", "--home", home]) == 0
    assert all(r["status"] == "active" for r in load_rules(Store(home).rules_path("orders")))
    assert main(["run", str(batches[1]), "--dataset", "orders", "--home", home]) == 0
    assert main(["run", str(batches[-1]), "--dataset", "orders", "--home", home]) == 1
    assert main(["history", "--home", home]) == 0
    out = capsys.readouterr().out
    assert "FAIL" in out and "Rule order_id.unique failed" in out


def test_cli_single_commands(batches, tmp_path):
    rules = tmp_path / "rules.yaml"
    assert main(["suggest-rules", str(batches[0]), "--out", str(rules), "--no-llm"]) == 0
    assert main(["check", str(batches[1]), "--rules", str(rules)]) == 0
    assert main(["check", str(batches[-1]), "--rules", str(rules), "--fail-on", "warning"]) == 1
    assert main(["drift", str(batches[0]), str(batches[-1])]) == 1
    assert main(["profile", str(batches[0]), "--out", str(tmp_path / "p.json")]) == 0
    assert main(["run", str(tmp_path / "missing.csv"), "--home", str(tmp_path / "h")]) == 2


class FakeGemini:
    def __init__(self, enrich_reply: str):
        self.enrich_reply = enrich_reply
        self.prompts = []

    def generate(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        return self.enrich_reply if json_mode else "**Verdict: block.** Upstream schema change."


def test_llm_enrichment_is_validated(clean):
    profile = profile_dataframe(clean, "orders")
    reply = """```json
    {"columns": [{"name": "amount", "semantic_type": "order_amount", "pii": false, "description": "value"},
                 {"name": "ghost", "semantic_type": "x"}],
     "rules": [{"type": "range", "column": "quantity", "params": {"min": 1, "max": 100}, "description": "qty"},
               {"type": "range", "column": "ghost", "params": {"min": 1}},
               {"type": "regex", "column": "sku", "params": {"pattern": "("}},
               {"type": "delete_table", "column": "sku"}]}
    ```"""
    client = FakeGemini(reply)
    annotations, rules = llm.enrich(client, profile)
    assert set(annotations) == {"amount"}
    assert [r["id"] for r in rules] == ["quantity.range.llm"]
    assert rules[0]["source"] == "llm" and rules[0]["status"] == "proposed"
    # raw values are not sent by default
    assert "user" not in client.prompts[0] and "example.com" not in client.prompts[0]


def test_pipeline_with_fake_llm(batches, tmp_path, monkeypatch):
    fake = FakeGemini('{"columns": [], "rules": []}')
    monkeypatch.setattr(llm, "get_client", lambda enabled=None: fake)
    store = Store(tmp_path / ".dqa")
    run_batch(batches[0], "orders", store)
    r = run_batch(batches[-1], "orders", store)
    assert r.llm_used and r.llm_summary.startswith("**Verdict")
    assert "not-an-email" not in fake.prompts[-1]  # sample values stay local


def test_get_client_without_key():
    assert llm.get_client(None) is None
    assert llm.get_client(True) is None
    assert llm.get_client(False) is None
