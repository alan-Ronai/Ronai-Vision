"""
Scenario Module - Integration with Armed Attack Demo Scenario

This module provides hooks and integration points for the AI service
to communicate with the backend scenario manager.

Includes:
- ScenarioHooks: Simple hooks for detection pipeline integration
- ScenarioRuleEngine: Full rule-based scenario state machine
"""

from .scenario_hooks import (
    ScenarioHooks,
    get_scenario_hooks,
    init_scenario_hooks,
    VehicleData,
    PersonData,
)

from .scenario_rule_engine import (
    ScenarioRuleEngine,
    ScenarioContext,
    get_scenario_rule_engine,
    init_scenario_rule_engine,
)

__all__ = [
    # Hooks (simple integration)
    "ScenarioHooks",
    "get_scenario_hooks",
    "init_scenario_hooks",
    "VehicleData",
    "PersonData",
    # Rule Engine (full state machine)
    "ScenarioRuleEngine",
    "ScenarioContext",
    "get_scenario_rule_engine",
    "init_scenario_rule_engine",
]
