import pytest

from guardian.templates import Template, TemplateError, config_hash, diff_settings, merge_overrides, parse_kv_args

T = Template.model_validate({
    "name": "cifar-resnet", "script": "train.py",
    "defaults": {"lr": 1e-3, "epochs": 30, "micro_batch": 256, "grad_accum": 1, "seed": 0},
    "overridable": {
        "lr": {"type": "float", "min": 1e-6, "max": 1e-1},
        "epochs": {"type": "int", "min": 1, "max": 300},
        "micro_batch": {"type": "int", "choices": [32, 64, 128, 256, 512]},
        "seed": {"type": "int", "min": 0, "max": 100000},
    },
})


def test_defaults_plus_coerced_overrides():
    cfg = merge_overrides(T, parse_kv_args(["lr=3e-4", "epochs=50"]))
    assert cfg["lr"] == pytest.approx(3e-4) and cfg["epochs"] == 50 and cfg["micro_batch"] == 256
    assert diff_settings(T.defaults, cfg) == ["epochs 50 (was 30)", "lr 0.0003 (was 0.001)"]


@pytest.mark.parametrize("ov,needle", [
    ({"lr": "5"}, "outside the allowed range"),
    ({"epochs": "0"}, "outside the allowed range"),
    ({"micro_batch": "100"}, "choose one of"),
    ({"epochs": "lots"}, "not a valid int"),
    ({"script": "/bin/rm"}, "not a setting you can change"),
])
def test_out_of_limits_is_refused_with_reason(ov, needle):
    with pytest.raises(TemplateError, match=needle):
        merge_overrides(T, ov)


def test_hash_ignores_execution_settings():
    a = merge_overrides(T, {})
    b = dict(a, micro_batch=128, grad_accum=2)
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(dict(a, lr=0.01))


def test_kv_parsing_rejects_garbage():
    with pytest.raises(TemplateError):
        parse_kv_args(["lr"])
