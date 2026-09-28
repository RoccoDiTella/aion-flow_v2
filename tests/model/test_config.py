"""M0: the run recipes are the paper's three runs, and nothing else validates."""

from __future__ import annotations

import pytest
import yaml

from aionflow_model.config import (
    TRAINING,
    ConfigError,
    Head,
    load_run,
    parse_run,
    recipe_path,
    with_lr_scale,
)
from aionflow_model.data import TARGETS

CONFIGS = "configs/{name}.yaml"


def test_the_three_reported_runs_differ_only_in_their_heads():
    runs = {name: load_run(CONFIGS.format(name=name))
            for name in ("marginals", "rates", "joint4")}
    assert [h.targets for h in runs["marginals"].heads] == [
        ("flux",), ("lx",), ("sfr",), ("mstar",), ("sfr", "mstar")]
    assert [h.targets for h in runs["rates"].heads] == [("rate_p2", "rate_p3")]
    assert [h.targets for h in runs["joint4"].heads] == [("rate_p2", "rate_p3", "sfr", "mstar")]
    assert runs["marginals"].targets == ("flux", "lx", "sfr", "mstar")
    assert runs["joint4"].targets == ("sfr", "mstar", "rate_p2", "rate_p3")
    assert runs["marginals"].head("sfr_mstar").is_joint
    assert not runs["marginals"].head("flux").is_joint
    assert runs["joint4"].heads[0].kinds == ("rate", "rate", "scalar", "scalar")


def test_the_optimizer_is_the_papers():
    assert TRAINING.betas == (0.95, 0.999)
    assert (TRAINING.lr_readout, TRAINING.lr_flow, TRAINING.lr_adapter) == (3e-4, 1e-3, 3e-5)
    assert (TRAINING.wd_readout, TRAINING.wd_adapter, TRAINING.wd_cls) == (1e-4, 0.1, 0.0)
    assert (TRAINING.batch_size, TRAINING.grad_clip) == (896, 5.0)
    assert (TRAINING.max_epochs, TRAINING.patience, TRAINING.seed) == (40, 5, 42)


@pytest.mark.parametrize("spec, message", [
    ({"heads": {"a": ["flux"]}}, "non-empty `name`"),
    ({"name": "r"}, "non-empty `heads`"),
    ({"name": "r", "heads": {}}, "non-empty `heads`"),
    ({"name": "r", "heads": {"a": []}}, "non-empty list"),
    ({"name": "r", "heads": {"a": "flux"}}, "non-empty list"),
    ({"name": "r", "heads": {"a": ["hardness"]}}, "unknown targets"),
    ({"name": "r", "heads": {"a": ["flux", "flux"]}}, "repeated target"),
    ({"name": "r", "heads": {"a": ["flux"]}, "lr": 1e-3}, "unknown recipe keys"),
])
def test_a_bad_recipe_is_refused(spec, message):
    with pytest.raises(ConfigError, match=message):
        parse_run(spec)


def test_load_run_round_trips_and_reports_a_missing_file(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump({"name": "r", "heads": {"a": ["flux", "lx"]}}))
    assert load_run(path).heads == (Head("a", ("flux", "lx")),)
    with pytest.raises(ConfigError, match="no run recipe"):
        load_run(tmp_path / "absent.yaml")


def test_every_target_names_columns_the_label_contract_has():
    from aionflow_data.labels import LABEL_COLUMNS
    for target in TARGETS.values():
        assert target.kind in ("scalar", "rate")
        assert set(target.columns) <= set(LABEL_COLUMNS), target.name


# ----------------------------------------------------------------------------- options

def test_a_recipe_may_scale_the_learning_rates_and_exclude_sf_dominated_galaxies():
    heads = {"flux": ["flux"]}
    run = parse_run({"name": "x", "heads": heads, "lr_scale": 0.5, "exclude_sf_dominated": True})
    assert run.lr_scale == 0.5 and run.exclude_sf_dominated is True
    plain = parse_run({"name": "x", "heads": heads})
    assert plain.lr_scale == 1.0 and plain.exclude_sf_dominated is False
    for bad in ({"lr_scale": 0}, {"lr_scale": -1}, {"lr_scale": True}, {"lr_scale": "half"},
                {"exclude_sf_dominated": "yes"}, {"exclude_sf_dominated": 1}):
        with pytest.raises(ConfigError):
            parse_run({"name": "x", "heads": heads, **bad})


def test_the_rate_scale_touches_the_three_learning_rates_and_nothing_else():
    half = with_lr_scale(TRAINING, 0.5)
    assert (half.lr_readout, half.lr_flow, half.lr_adapter) == pytest.approx((1.5e-4, 5e-4, 1.5e-5))
    assert (half.batch_size, half.wd_adapter, half.patience, half.seed) == (
        TRAINING.batch_size, TRAINING.wd_adapter, TRAINING.patience, TRAINING.seed)


def test_only_joint4_halves_its_rates_and_withholds_sf_dominated_galaxies():
    joint4 = load_run(recipe_path("joint4"))
    assert joint4.lr_scale == 0.5 and joint4.exclude_sf_dominated
    for name in ("marginals", "rates", "baseline", "pooling"):
        other = load_run(recipe_path(name))
        assert other.lr_scale == 1.0 and not other.exclude_sf_dominated, name
