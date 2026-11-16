"""Scheduler / config findings (fake GPUs, no processes are started: _launch is stubbed)."""
import pytest

from conftest import outbox, queue
from conftest import make_cfg, write_toy_template

V100 = {"name": "Tesla V100-PCIE-16GB", "mem_gb": 16, "slowdown_c": 87}


def stub_launch(d):
    started = []

    def fake(job, gpus):
        started.append((job["id"], list(gpus)))
        d.db.update_job(job["id"], state="running", gpus=",".join(map(str, gpus)))

    d.runner._launch = fake
    return started


# ---- B12: a job that cannot start (quarantined card / card removed) blocks the whole queue silently
def test_b12_wide_job_behind_quarantine_must_not_freeze_queue_silently(make_daemon, clock, tmp_path):
    write_toy_template(tmp_path, name="wide", gpus=2)
    d = make_daemon(clock=clock)
    started = stub_launch(d)
    wide = queue(d, template="wide")
    small1, small2 = queue(d), queue(d)
    d.db.set_gpu_state(1, quarantined=1, quarantine_reason="test: emergency last night")
    for _ in range(1800):  # one hour of scheduler ticks
        clock.advance(2)
        d.collector.collect_once()
        d.runner.tick()
        d.watchdog.tick()
    told = [t for t in outbox(d) if wide in t or "queue" in t.lower()]
    assert started or told, (
        f"GPU 0 idle for an hour, 3 jobs queued ({wide} needs 2 GPUs, {small1}/{small2} need 1): "
        f"nothing started and nothing was said. outbox={outbox(d)}")


# ---- B16: config validation gaps --------------------------------------------------------
def test_b16_duplicate_training_index_rejected(tmp_path):
    write_toy_template(tmp_path)
    with pytest.raises(ValueError):
        make_cfg(tmp_path, gpus={"training": [0, 0]}).resolve_gpus([0, 1, 2])


def test_b16_duplicate_training_index_is_refused_before_it_can_give_one_gpu_twice(tmp_path):
    with pytest.raises(ValueError, match="same GPU twice"):
        make_cfg(tmp_path, gpus={"training": [0, 0]})


def test_b16_a_card_both_pinned_for_training_and_excluded_is_refused(tmp_path):
    with pytest.raises(ValueError, match="exclude"):
        make_cfg(tmp_path, gpus={"training": [0, 1], "exclude": [1]})


@pytest.mark.parametrize("section,key,value", [
    ("thermal", "smoothing_samples", 0), ("telemetry", "interval_s", 0), ("runner", "poll_s", -1),
    ("thermal", "pause_margin_c", -5), ("runner", "free_mem_mb", -1), ("stall", "no_progress_s", 0),
    ("telegram", "quiet_hours", ["2300", "0700"]), ("telegram", "digest_time", "8am"),
])
def test_b16_nonsense_values_rejected_at_load(tmp_path, section, key, value):
    with pytest.raises(ValueError):
        make_cfg(tmp_path, **{section: {key: value}})


@pytest.mark.parametrize("section,key,value", [
    ("thermal", "smoothing_samples", 0), ("telegram", "digest_time", "8am"), ("telegram", "quiet_hours", ["2300", "0700"]),
])
def test_b16_settings_that_would_switch_off_protection_or_messages_never_reach_a_running_agent(tmp_path, section, key, value):
    with pytest.raises(ValueError):
        make_cfg(tmp_path, **{section: {key: value}})


def test_b16_bad_quiet_hours_cannot_lose_a_message_even_if_they_get_past_the_loader(make_daemon, clock):
    d = make_daemon(clock=clock)
    d.cfg.telegram.quiet_hours = ["2300", "0700"]  # as if an older config had been loaded unchecked
    d.notifier.send("▶ j_0001 resumed")
    assert "▶ j_0001 resumed" in outbox(d)
