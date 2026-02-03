"""Tests for block-level variable transformations."""

from equilibrium.model import Model, ModelBlock


def test_block_transformations_apply_to_model_rules():
    block = ModelBlock(
        params={"c": 1.0},
        rules={
            "intermediate": [
                ("x", "c + 1"),
                ("y", "x_NEXT + x"),
            ]
        },
    )
    block.transform_variables(["x"], "np.log", "np.exp", prefix="log")

    model = Model(inner_functions={})
    model.add_block(block)
    model._update_rules()

    rules = model.rules["intermediate"]
    assert "log_x" in rules
    assert rules["log_x"] == "np.log(c + 1)"
    assert rules["y"] == "np.exp(log_x_NEXT) + np.exp(log_x)"


def test_block_transformations_respect_suffix():
    block = ModelBlock(
        params={"a": 1.0},
        rules={
            "intermediate": [
                ("x", "a + 1"),
                ("y", "x + 2"),
            ]
        },
    )
    block.transform_variables(["x"], "np.log", "np.exp", prefix="log")

    model = Model(inner_functions={})
    model.add_block(block, suffix="_firm")
    model._update_rules()

    rules = model.rules["intermediate"]
    assert "log_x_firm" in rules
    assert rules["log_x_firm"] == "np.log(a + 1)"
    assert rules["y_firm"] == "np.exp(log_x_firm) + 2"


def test_block_transformations_with_rename_and_suffix():
    block = ModelBlock(
        params={"a_AGENT": 1.0},
        rules={
            "intermediate": [
                ("x_AGENT", "a_AGENT + 1"),
                ("y_AGENT", "x_AGENT + 2"),
            ]
        },
    )
    block.transform_variables(["x_AGENT"], "np.log", "np.exp", prefix="log")

    model = Model(inner_functions={})
    model.add_block(block, rename={"AGENT": "borrower"}, suffix="_firm")
    model._update_rules()

    rules = model.rules["intermediate"]
    assert "log_x_borrower_firm" in rules
    assert rules["log_x_borrower_firm"] == "np.log(a_borrower + 1)"
    assert rules["y_borrower_firm"] == "np.exp(log_x_borrower_firm) + 2"
