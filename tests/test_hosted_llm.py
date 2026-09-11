import json

import pytest

from app import bedrock_engine, hosted_llm, openai_compat_engine
from app.engine import AdjudicationEngine
from app.bedrock_engine import BedrockEngine
from app.openai_compat_engine import OpenAICompatEngine


_SEPTIC_PAYLOAD = {
    "clinical_rationale": "Chart documents Klebsiella urosepsis with shock requiring Levophed.",
    "primary_criteria_met": ["infection", "vasopressors"],
    "key_evidence": [{
        "note_id": 40003,
        "note_date": "2026-03-08",
        "evidence_quote": "Severe Sepsis and Septic Shock secondary to Klebsiella urosepsis",
        "interpretation": "Discharge diagnosis",
    }],
    "phenotype_status": "CONFIRMED_POSITIVE",
    "condition_present": True,
    "confidence_score": 0.83,
}

_RECORD = {
    "person_id": 1001,
    "visit_occurrence_id": 20001,
    "visit_start_date": "2026-03-01",
    "visit_end_date": "2026-03-08",
    "notes_formatted_text": "Severe Sepsis and Septic Shock secondary to Klebsiella urosepsis",
}


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.contents:
            raise TimeoutError("deadline")
        return _Resp(self.contents.pop(0))


class _Chat:
    def __init__(self, contents):
        self.completions = _Completions(contents)


class _FakeOpenAI:
    def __init__(self, *contents):
        self.chat = _Chat(contents)


class _FakeBedrock:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if not self.texts:
            raise TimeoutError("deadline")
        text = self.texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return {"output": {"message": {"content": [{"text": text}]}}}


def test_https_endpoint_policy(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_OPENAI_ALLOW_HTTP", raising=False)
    with pytest.raises(ValueError, match="not set"):
        hosted_llm.validate_https_endpoint(None, env_name="ROSEGOLD_OPENAI_BASE_URL")
    with pytest.raises(ValueError, match="plain http"):
        hosted_llm.validate_https_endpoint(
            "http://10.0.0.5:443/v1", env_name="ROSEGOLD_OPENAI_BASE_URL"
        )
    for bad in ("ftp://x/y", "adb.example", "https://"):
        with pytest.raises(ValueError):
            hosted_llm.validate_https_endpoint(bad, env_name="ROSEGOLD_OPENAI_BASE_URL")
    assert hosted_llm.validate_https_endpoint(
        "https://adb.example/serving-endpoints", env_name="ROSEGOLD_OPENAI_BASE_URL"
    )
    assert hosted_llm.validate_https_endpoint(
        "http://127.0.0.1:8001/v1", env_name="ROSEGOLD_OPENAI_BASE_URL"
    )
    monkeypatch.setenv("ROSEGOLD_OPENAI_ALLOW_HTTP", "1")
    assert hosted_llm.validate_https_endpoint(
        "http://10.0.0.5:443/v1", env_name="ROSEGOLD_OPENAI_BASE_URL"
    )


def test_mantle_url_from_region(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ROSEGOLD_MANTLE_PATH", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert hosted_llm.mantle_base_url() == "https://bedrock-mantle.us-west-2.api.aws/v1"
    monkeypatch.setenv("ROSEGOLD_MANTLE_PATH", "openai/v1")
    assert hosted_llm.mantle_base_url("us-east-1") == "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    with pytest.raises(ValueError, match="Invalid AWS region"):
        hosted_llm.resolve_aws_region("https://evil.example")
    with pytest.raises(ValueError, match="ROSEGOLD_MANTLE_PATH"):
        monkeypatch.setenv("ROSEGOLD_MANTLE_PATH", "v2")
        hosted_llm.mantle_base_url()


def test_mantle_requires_token(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "mantle")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("ROSEGOLD_OPENAI_MODEL", "openai.gpt-oss-120b")
    for name in (
        "ROSEGOLD_OPENAI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "DATABRICKS_TOKEN",
        "DATABRICKS_API_TOKEN",
        "OPENAI_API_KEY",
        "ROSEGOLD_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK"):
        OpenAICompatEngine(client=_FakeOpenAI(json.dumps(_SEPTIC_PAYLOAD)))


def test_mantle_uses_client_and_tags_backend():
    client = _FakeOpenAI(json.dumps(_SEPTIC_PAYLOAD))
    engine = OpenAICompatEngine(
        model_name="openai.gpt-oss-120b",
        base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
        api_key="bedrock-key",
        client=client,
        backend_tag="mantle",
    )
    rows = engine.adjudicate_batch([_RECORD], "Sepsis / Septic Shock", "Sepsis-3")
    assert rows[0]["inference_backend"] == "mantle:openai.gpt-oss-120b"
    assert engine.base_url == "https://bedrock-mantle.us-west-2.api.aws/v1"
    assert client.chat.completions.calls[0]["model"] == "openai.gpt-oss-120b"


def test_databricks_host_appends_serving_endpoints(monkeypatch):
    monkeypatch.delenv("ROSEGOLD_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb.example.net")
    assert openai_compat_engine.resolve_base_url() == "https://adb.example.net/serving-endpoints"
    monkeypatch.setenv("DATABRICKS_HOST", "adb.example.net")
    assert openai_compat_engine.resolve_base_url().startswith("https://adb.example.net/serving-endpoints")


def test_databricks_requires_token(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "databricks")
    monkeypatch.setenv("ROSEGOLD_OPENAI_BASE_URL", "https://adb.example/serving-endpoints")
    monkeypatch.setenv("ROSEGOLD_OPENAI_MODEL", "site-llm")
    for name in ("ROSEGOLD_OPENAI_API_KEY", "DATABRICKS_TOKEN", "DATABRICKS_API_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="DATABRICKS_TOKEN"):
        OpenAICompatEngine(client=_FakeOpenAI(json.dumps(_SEPTIC_PAYLOAD)))


def test_openai_compat_uses_client_and_tags_backend(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "databricks")
    client = _FakeOpenAI(json.dumps(_SEPTIC_PAYLOAD))
    engine = OpenAICompatEngine(
        model_name="site-approved-llm",
        base_url="https://adb.example/serving-endpoints",
        api_key="dapi-test",
        client=client,
        backend_tag="databricks",
    )
    rows = engine.adjudicate_batch([_RECORD], "Sepsis / Septic Shock", "Sepsis-3")
    assert rows[0]["inference_backend"] == "databricks:site-approved-llm"
    assert rows[0]["clinical_rationale"].startswith("Chart documents Klebsiella")
    assert rows[0]["confidence_score"] == 0.83
    assert client.chat.completions.calls[0]["model"] == "site-approved-llm"
    assert client.chat.completions.calls[0]["temperature"] == 0.0


def test_openai_compat_one_bad_response_does_not_abort_batch():
    client = _FakeOpenAI("not-json", json.dumps(_SEPTIC_PAYLOAD))
    engine = OpenAICompatEngine(
        model_name="site-llm",
        base_url="https://adb.example/serving-endpoints",
        api_key="dapi-test",
        client=client,
        backend_tag="databricks",
    )
    second = dict(_RECORD, visit_occurrence_id=20002)
    rows = engine.adjudicate_batch([_RECORD, second], "Sepsis / Septic Shock", "Sepsis-3")
    assert rows[0]["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert rows[0]["inference_backend"] == "databricks:site-llm"
    assert rows[1]["phenotype_status"] == "CONFIRMED_POSITIVE"
    assert rows[1]["visit_occurrence_id"] == 20002


def test_bedrock_uses_converse_and_tags_backend():
    client = _FakeBedrock(json.dumps(_SEPTIC_PAYLOAD))
    engine = BedrockEngine(model_name="anthropic.claude-test", region="us-west-2", client=client)
    rows = engine.adjudicate_batch([_RECORD], "Sepsis / Septic Shock", "Sepsis-3")
    assert rows[0]["inference_backend"] == "bedrock:anthropic.claude-test"
    assert rows[0]["condition_present"] is True
    assert client.calls[0]["modelId"] == "anthropic.claude-test"
    assert client.calls[0]["inferenceConfig"]["temperature"] == 0.0


def test_bedrock_one_bad_response_does_not_abort_batch():
    client = _FakeBedrock(TimeoutError("deadline"), json.dumps(_SEPTIC_PAYLOAD))
    engine = BedrockEngine(model_name="anthropic.claude-test", client=client)
    second = dict(_RECORD, visit_occurrence_id=20002)
    rows = engine.adjudicate_batch([_RECORD, second], "Sepsis / Septic Shock", "Sepsis-3")
    assert rows[0]["phenotype_status"] == "INDETERMINATE_INSUFFICIENT_DATA"
    assert rows[1]["phenotype_status"] == "CONFIRMED_POSITIVE"


def test_redact_secret_and_describe_config(monkeypatch):
    assert hosted_llm.redact_secret("") == ""
    assert hosted_llm.redact_secret("abcd") == "****"
    redacted = hosted_llm.redact_secret("dapiXXXXXXXXYY")
    assert "dapi" in redacted
    assert "YY" in redacted
    assert "XXXX" not in redacted
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "databricks")
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb.example.net")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapiXXXXXXXXYY")
    monkeypatch.setenv("ROSEGOLD_OPENAI_MODEL", "site-llm")
    cfg = hosted_llm.describe_hosted_config()
    assert cfg["docker_required"] is False
    assert cfg["singularity_required"] is False
    assert cfg["api_key_present"] is True
    assert "dapiXXXXXXXXYY" not in json.dumps(cfg)


def test_engine_databricks_skips_local_vllm(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "databricks")
    monkeypatch.setenv("ROSEGOLD_OPENAI_BASE_URL", "https://adb.example/serving-endpoints")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-test")
    monkeypatch.setenv("ROSEGOLD_OPENAI_MODEL", "site-llm")
    monkeypatch.delenv("ROSEGOLD_ALLOW_MOCK", raising=False)

    class _FakeHosted:
        def __init__(self, *a, **k):
            self.model_name = "site-llm"
            self.backend_tag = "databricks"

        def adjudicate_batch(self, records, target_condition, clinical_criteria):
            return [{
                "person_id": records[0]["person_id"],
                "visit_occurrence_id": records[0]["visit_occurrence_id"],
                "condition_present": True,
                "phenotype_status": "CONFIRMED_POSITIVE",
                "confidence_score": 0.9,
                "primary_criteria_met": [],
                "key_evidence": [],
                "clinical_rationale": "ok",
                "inference_backend": "databricks:site-llm",
            }]

    monkeypatch.setattr(openai_compat_engine, "OpenAICompatEngine", _FakeHosted)
    engine = AdjudicationEngine(model_name="auto")
    engine.is_gpu = True
    status = engine.backend_status(init=True)
    assert status["backend"] == "databricks"
    assert status["llm_real"] is True
    assert engine.llm is None
    assert engine.has_real_backend()
    out = engine.adjudicate_single(_RECORD, "Sepsis / Septic Shock", "Sepsis-3")
    assert out["inference_backend"] == "databricks:site-llm"


def test_engine_bedrock_defaults_to_mantle(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "bedrock")
    monkeypatch.delenv("ROSEGOLD_BEDROCK_API", raising=False)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    monkeypatch.setenv("ROSEGOLD_OPENAI_MODEL", "openai.gpt-oss-120b")
    monkeypatch.delenv("ROSEGOLD_ALLOW_MOCK", raising=False)

    class _FakeMantle:
        def __init__(self, *a, **k):
            self.model_name = "openai.gpt-oss-120b"
            self.backend_tag = "mantle"

        def adjudicate_batch(self, records, target_condition, clinical_criteria):
            return [{
                "person_id": records[0]["person_id"],
                "visit_occurrence_id": records[0]["visit_occurrence_id"],
                "condition_present": True,
                "phenotype_status": "CONFIRMED_POSITIVE",
                "confidence_score": 0.9,
                "primary_criteria_met": [],
                "key_evidence": [],
                "clinical_rationale": "ok",
                "inference_backend": "mantle:openai.gpt-oss-120b",
            }]

    monkeypatch.setattr(openai_compat_engine, "OpenAICompatEngine", _FakeMantle)
    engine = AdjudicationEngine(model_name="auto")
    engine.is_gpu = True
    status = engine.backend_status(init=True)
    assert status["backend"] == "mantle"
    assert engine.bedrock_engine is None
    assert engine.llm is None
    assert engine.adjudicate_single(_RECORD, "Sepsis / Septic Shock", "Sepsis-3")["inference_backend"].startswith("mantle:")


def test_engine_bedrock_converse_when_requested(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "bedrock")
    monkeypatch.setenv("ROSEGOLD_BEDROCK_API", "converse")
    monkeypatch.delenv("ROSEGOLD_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ROSEGOLD_BEDROCK_MODEL", "anthropic.claude-test")
    monkeypatch.delenv("ROSEGOLD_ALLOW_MOCK", raising=False)

    class _FakeNative:
        def __init__(self, *a, **k):
            self.model_name = "anthropic.claude-test"

        def adjudicate_batch(self, records, target_condition, clinical_criteria):
            return [{
                "person_id": records[0]["person_id"],
                "visit_occurrence_id": records[0]["visit_occurrence_id"],
                "condition_present": False,
                "phenotype_status": "CONFIRMED_NEGATIVE",
                "confidence_score": 0.7,
                "primary_criteria_met": [],
                "key_evidence": [],
                "clinical_rationale": "ok",
                "inference_backend": "bedrock:anthropic.claude-test",
            }]

    monkeypatch.setattr(bedrock_engine, "BedrockEngine", _FakeNative)
    engine = AdjudicationEngine(model_name="auto")
    status = engine.backend_status(init=True)
    assert status["backend"] == "bedrock"
    assert engine.hosted_engine is None
    assert engine.bedrock_engine is not None
