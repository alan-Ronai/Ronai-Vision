"""
Event Rules Engine Module

Provides the rule evaluation engine for processing events
based on configurable rules with conditions, pipeline, and actions.
"""

from .rule_engine import RuleEngine, RuleContext, get_rule_engine

__all__ = ['RuleEngine', 'RuleContext', 'get_rule_engine']
