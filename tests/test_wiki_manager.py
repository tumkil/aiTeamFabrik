"""Tests for the WikiManager module."""

import pytest
from factory.core.wiki_manager import WikiManager


class TestWikiManagerFormatTemplate:
    """Tests for WikiManager.format_template."""

    def setup_method(self):
        self.wiki = WikiManager()

    def test_format_template_with_curly_braces_in_kwargs(self):
        """format_template should not raise KeyError when kwargs values contain curly braces."""
        result = self.wiki.format_template(
            "architecture_overview",
            architect_plan="some plan with {curly} braces",
            po_comments="{another} brace",
        )
        assert "some plan with {curly} braces" in result
        assert "{another} brace" in result
        assert "*Last updated:" in result

    def test_format_template_default_date(self):
        """format_template should inject today's date when 'date' is not provided."""
        result = self.wiki.format_template("architecture_overview")
        assert "*Last updated:" in result

    def test_format_template_custom_date(self):
        """format_template should use the provided date when 'date' is given."""
        result = self.wiki.format_template("architecture_overview", date="2024-01-15")
        assert "*Last updated: 2024-01-15*" in result

    def test_format_template_no_extra_append_for_date(self):
        """format_template should not append 'date' as a key-value pair."""
        result = self.wiki.format_template("architecture_overview", date="2024-01-15")
        assert "\n\ndate:" not in result

    def test_format_template_appends_extra_kwargs(self):
        """format_template should append extra kwargs to the template."""
        result = self.wiki.format_template(
            "architecture_overview",
            extra_key="extra_value",
        )
        assert "\n\nextra_key: extra_value" in result
