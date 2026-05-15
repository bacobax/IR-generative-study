from src.core.registry import REGISTRIES, Registry


def test_existing_and_adapter_registries_exist() -> None:
    for attr in ("model_builder", "trainer", "sampler", "guidance", "conditioning"):
        assert hasattr(REGISTRIES, attr)
    for attr in ("model_adapter", "dataset_adapter", "task_adapter", "artifact_loader"):
        assert hasattr(REGISTRIES, attr)


def test_registry_behavior_is_unchanged() -> None:
    registry = Registry("unit")

    @registry.register("thing", default=True)
    def thing():
        return "ok"

    assert registry["thing"]() == "ok"
    assert registry.get()() == "ok"
    assert registry.list() == ["thing"]

    try:
        registry.register("thing")(lambda: "duplicate")
    except ValueError as exc:
        assert "[unit] 'thing' is already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")


def test_registry_summary_includes_adapter_registries() -> None:
    summary = REGISTRIES.summary()
    for name in (
        "model_builder",
        "trainer",
        "sampler",
        "guidance",
        "conditioning",
        "model_adapter",
        "dataset_adapter",
        "task_adapter",
        "artifact_loader",
    ):
        assert f"  {name}:" in summary
