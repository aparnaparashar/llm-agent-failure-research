import pytest
from failure_taxonomy.loader import FailureTaxonomy, get_taxonomy


def test_taxonomy_loading():
    tax = get_taxonomy()
    assert isinstance(tax, FailureTaxonomy)
    modules = tax.get_modules()
    assert len(modules) >= 5
    assert "reflection" in modules or "Reflection" in [m.title() for m in modules]
    assert "action" in modules or "Action" in [m.title() for m in modules]
    assert "planning" in modules or "Planning" in [m.title() for m in modules]
    assert "memory" in modules or "Memory" in [m.title() for m in modules]
    assert "system" in modules or "System" in [m.title() for m in modules]


def test_injectability_classification():
    tax = get_taxonomy()
    # Check that tool execution error is injectable
    is_inj = tax.is_injectable("system", "tool_execution_error")
    assert is_inj is True

    # Check non-existent failure returns False
    assert tax.is_injectable("system", "non_existent_failure_xyz") is False


def test_failure_validation():
    tax = get_taxonomy()
    assert tax.validate_failure("system", "tool_execution_error") is True
    assert tax.validate_failure("system", "fake_failure_123") is False
