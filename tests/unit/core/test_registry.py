"""Unit tests for ovtas.registry.Registry."""

from __future__ import annotations

import pytest

from ovtas.registry import Registry, RegistryError


@pytest.mark.unit
def test_register_and_get():
    reg = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    # Lookups are case-insensitive.
    assert reg.get("FOO") is Foo


@pytest.mark.unit
def test_duplicate_registration_raises():
    reg = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(RegistryError):
        @reg.register("foo")
        class Bar:
            pass


@pytest.mark.unit
def test_unknown_lookup_raises_with_helpful_message():
    reg = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    with pytest.raises(RegistryError, match="Unknown widget 'bar'"):
        reg.get("bar")


@pytest.mark.unit
def test_build_instantiates_with_args():
    reg = Registry("widget")

    @reg.register("adder")
    class Adder:
        def __init__(self, a, b):
            self.total = a + b

    obj = reg.build("adder", 2, 3)
    assert obj.total == 5


@pytest.mark.unit
def test_contains_and_len_and_iter():
    reg = Registry("widget")

    @reg.register("a")
    class A:
        pass

    @reg.register("b")
    class B:
        pass

    assert "a" in reg
    assert "c" not in reg
    assert len(reg) == 2
    assert list(reg) == ["a", "b"]
