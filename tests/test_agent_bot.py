"""The command router, confirmations, allowlist and the brain (with a fake llama-server)."""

import json

import httpx
import pytest

from conftest import queue
from guardian.actions import Agent
from guardian.brain import Brain, numbers_grounded

USER = 42


@pytest.fixture
def d(make_daemon, clock):
    return make_daemon(clock=clock)


def fake_llm(*answers):
    """A transport that replays model answers; an Exception answer simulates the server being down."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        ans = answers[min(len(calls) - 1, len(answers) - 1)]
        if isinstance(ans, Exception):
            raise ans
        content = ans if isinstance(ans, str) else json.dumps(ans)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.MockTransport(handler), calls


def with_brain(d, *answers):
    transport, calls = fake_llm(*answers)
    d.cfg.brain.enabled = True
    agent = Agent(d.cfg, d.db, d.runner, d.reports, d.notifier, brain=Brain(d.cfg.brain, transport=transport),
                  clock=d.clock)
    return agent, calls


def test_strangers_and_groups_get_no_reply(d):
    assert d.agent.handle_text(999, "/status") is None
    assert d.agent.handle_text(USER, "/status", chat_type="group") is None
    assert d.db.one("SELECT COUNT(*) AS n FROM events WHERE kind='telegram_denied'")["n"] == 2
    assert d.agent.handle_callback(999, "ok:whatever") is None


def test_status_and_help(d):
    [r] = d.agent.handle_text(USER, "/status")
    assert r.text.startswith("🖥 Status") and "V100 #0" in r.text and "V100 #1" in r.text and "P4000" in r.text
    [r] = d.agent.handle_text(USER, "/help")
    assert "/rerun" in r.text


def test_run_out_of_limits_is_refused_with_reason(d):
    [r] = d.agent.handle_text(USER, "/run toy-mlp lr=5")
    assert "outside the allowed range" in r.text and r.buttons is None
    assert d.db.queued_jobs() == []


def test_run_needs_confirm_and_code_is_one_time(d):
    [r] = d.agent.handle_text(USER, "/run toy-mlp lr=3e-4 epochs=5")
    assert r.text.startswith("Queue toy-mlp?") and "lr 0.0003 (was 0.003)" in r.text
    assert "is free" in r.text
    assert d.db.queued_jobs() == []  # nothing until the tap
    ok = r.buttons[0][1]
    done = d.agent.handle_callback(USER, ok)
    assert done.text.startswith("⏳ Queued j_0001")
    assert len(d.db.queued_jobs()) == 1
    again = d.agent.handle_callback(USER, ok)
    assert "already been used" in again.text and len(d.db.queued_jobs()) == 1
    audit = d.db.one("SELECT * FROM audit WHERE action='queue_run'")
    assert audit["confirmed"] == 1 and audit["telegram_user"] == USER


def test_confirmation_expires_after_five_minutes(d, clock):
    [r] = d.agent.handle_text(USER, "/run toy-mlp")
    clock.advance(301)
    res = d.agent.handle_callback(USER, r.buttons[0][1])
    assert "expired" in res.text and d.db.queued_jobs() == []


def test_cancel_button_changes_nothing(d):
    [r] = d.agent.handle_text(USER, "/run toy-mlp")
    res = d.agent.handle_callback(USER, r.buttons[1][1])
    assert res.text.startswith("Cancelled") and d.db.queued_jobs() == []


def test_rerun_shows_changes_against_parent(d):
    parent = queue(d, {"lr": 1e-3})
    d.db.update_job(parent, state="completed")
    [r] = d.agent.handle_text(USER, f"/rerun {parent} lr=5e-4 seed=1")
    assert "lr 0.0005 (was 0.001)" in r.text and "seed 1 (was 0)" in r.text
    d.agent.handle_callback(USER, r.buttons[0][1])
    child = d.db.queued_jobs()[-1]
    assert child["parent_job"] == parent


def test_job_ids_accept_short_forms(d):
    jid = queue(d)
    [r] = d.agent.handle_text(USER, "/job 1")
    assert r.text.startswith(jid)


def test_hold_and_clear(d):
    [r] = d.agent.handle_text(USER, "/hold")
    assert d.db.kv_get("hold") == "1"
    d.agent.handle_text(USER, "/unhold")
    assert d.db.kv_get("hold") is None
    [r] = d.agent.handle_text(USER, "/clear 1")
    assert "not quarantined" in r.text
    d.db.set_gpu_state(1, quarantined=1, quarantine_reason="ECC")
    [r] = d.agent.handle_text(USER, "/clear 1")
    d.agent.handle_callback(USER, r.buttons[0][1])
    assert d.db.gpu_state(1)["quarantined"] == 0


# ---- brain -------------------------------------------------------------------------------
def test_free_text_maps_to_confirmed_rerun(d):
    parent = queue(d)
    d.db.update_job(parent, state="completed")
    agent, calls = with_brain(d, {"action": "rerun", "args": {"job_id": parent, "overrides": {"lr": 5e-4, "seed": 1}},
                                  "reply": "Rerun with lr 5e-4 and seed 1"})
    [r] = agent.handle_text(USER, "nice, do it again with lr 5e-4 and seed 1")
    assert r.buttons and "lr 0.0005" in r.text
    body = calls[0]
    assert body["response_format"]["type"] == "json_schema"  # output constrained to the menu
    prompt = body["messages"][0]["content"]
    assert parent in prompt and "toy-mlp" in prompt  # recent jobs and templates given, IDs not invented
    assert "Last 30" not in prompt  # logs never go to the model


def test_shell_request_maps_to_unsupported(d):
    agent, _ = with_brain(d, {"action": "unsupported", "args": {"reason": "I can't run shell commands"},
                              "reply": ""})
    [r] = agent.handle_text(USER, "run rm -rf on the server")
    assert r.text.startswith("AI: I can't do that") and r.buttons is None


def test_invalid_then_retry_once_then_give_up_with_reason(d):
    bad = {"action": "queue_run", "args": {"template": "toy-mlp", "overrides": {"lr": 5}}, "reply": ""}
    agent, calls = with_brain(d, bad, bad)
    [r] = agent.handle_text(USER, "toy run with lr 5")
    assert len(calls) == 2  # one retry, with the error fed back
    assert "rejected" in calls[1]["messages"][-1]["content"]
    assert "outside the allowed range" in r.text and "/run toy-mlp" in r.text


def test_made_up_job_id_is_rejected(d):
    agent, _ = with_brain(d, {"action": "get_job", "args": {"job_id": "j_0999"}, "reply": ""})
    [r] = agent.handle_text(USER, "how is my run")
    assert "couldn't map" in r.text


def test_brain_offline_but_slash_commands_work(d):
    agent, _ = with_brain(d, httpx.ConnectError("refused"))
    [r] = agent.handle_text(USER, "how are things")
    assert r.text.startswith("AI offline; slash commands still work")
    [r] = agent.handle_text(USER, "/status")
    assert r.text.startswith("🖥 Status")


def test_summaries_must_not_invent_numbers():
    facts = json.dumps({"best": {"val_loss": 0.412, "epoch": 27}, "duration": "1 h 52 m"})
    assert numbers_grounded("Best val_loss 0.412 at epoch 27 after 1 h 52 m.", facts)
    assert not numbers_grounded("Val loss improved by 12% to 0.412.", facts)


def test_cli_store_sees_daemon_readings(d):
    from guardian.telemetry import TelemetryStore

    st = TelemetryStore.from_db(d.db)
    assert [st.label(g) for g in (0, 1, 2)] == ["V100 #0", "V100 #1", "P4000"]
    assert st.latest(1)["mem_total"] == 16 * 2**30
