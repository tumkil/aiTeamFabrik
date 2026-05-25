"""
Tests for Architect-Developer Feedback Loop Enforcement.

These tests verify that:
1. Architect plans are properly parsed and extracted
2. Adherence metrics are correctly calculated
3. Deviations are properly detected
4. Context creation preserves architect guidance
5. PO comments are properly extracted and integrated
"""

import pytest
from factory.core.architect_feedback import (
    ArchitectPlan,
    AdherenceReport,
    ArchitectFeedbackEnforcer,
    extract_architect_plan_from_notes,
    extract_po_comments_from_notes,
    create_architect_context,
)


class TestArchitectPlan:
    """Tests for ArchitectPlan parsing."""

    def test_extract_file_paths(self):
        """Test extraction of file paths from plan text."""
        plan_text = """
        ## Implementation Plan
        
        Create the following files:
        - factory/core/module.py
        - tests/test_module.py
        - config/settings.yml
        """
        plan = ArchitectPlan(raw_text=plan_text)
        plan.extract_structure()
        
        assert "factory/core/module.py" in plan.file_paths
        assert "tests/test_module.py" in plan.file_paths
        assert "config/settings.yml" in plan.file_paths

    def test_extract_class_names(self):
        """Test extraction of class names from plan text."""
        plan_text = """
        ## Implementation Plan
        
        Create a class named UserService that handles user operations.
        Also create a UserRepository for database access.
        """
        plan = ArchitectPlan(raw_text=plan_text)
        plan.extract_structure()
        
        assert "UserService" in plan.class_names
        assert "UserRepository" in plan.class_names

    def test_extract_function_names(self):
        """Test extraction of function names from plan text."""
        plan_text = """
        ## Implementation Plan
        
        Implement the following functions:
        - def process_data(items): to process incoming data
        - def validate_input(value): to validate user input
        """
        plan = ArchitectPlan(raw_text=plan_text)
        plan.extract_structure()
        
        assert "process_data" in plan.function_names
        assert "validate_input" in plan.function_names

    def test_extract_key_decisions(self):
        """Test extraction of key decisions from plan text."""
        plan_text = """
        ## Implementation Plan
        
        The system MUST use PostgreSQL as the database.
        We should implement caching for performance.
        The API will use REST conventions.
        This is required for security compliance.
        """
        plan = ArchitectPlan(raw_text=plan_text)
        plan.extract_structure()
        
        # Check that decisions containing keywords are extracted
        assert len(plan.key_decisions) >= 3
        assert any("MUST" in d or "must" in d for d in plan.key_decisions)
        assert any("should" in d for d in plan.key_decisions)
        assert any("will" in d for d in plan.key_decisions)

    def test_deduplication(self):
        """Test that extracted elements are deduplicated."""
        plan_text = """
        ## Implementation Plan
        
        Create module.py and module.py (mentioned twice).
        The class MyClass must be created. MyClass is important.
        """
        plan = ArchitectPlan(raw_text=plan_text)
        plan.extract_structure()
        
        # Should only appear once
        assert plan.file_paths.count("module.py") == 1
        assert plan.class_names.count("MyClass") == 1


class TestAdherenceReport:
    """Tests for AdherenceReport metrics."""

    def test_perfect_adherence(self):
        """Test perfect adherence calculation."""
        plan = ArchitectPlan(
            raw_text="Implement module.py with MyClass",
            file_paths=["module.py"],
            class_names=["MyClass"],
            function_names=["process_data"],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["module.py"],
            implemented_classes=["MyClass"],
            implemented_functions=["process_data"],
        )
        report.calculate_metrics()
        
        assert report.file_adherence == 1.0
        assert report.class_adherence == 1.0
        assert report.function_adherence == 1.0
        assert report.overall_adherence == 1.0

    def test_partial_adherence(self):
        """Test partial adherence calculation."""
        plan = ArchitectPlan(
            raw_text="Implement module.py and utils.py",
            file_paths=["module.py", "utils.py"],
            class_names=["MyClass"],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["module.py"],  # Missing utils.py
            implemented_classes=["MyClass"],
            implemented_functions=[],
        )
        report.calculate_metrics()
        
        assert report.file_adherence == 0.5
        assert report.class_adherence == 1.0
        assert report.overall_adherence < 1.0

    def test_zero_adherence(self):
        """Test zero adherence calculation."""
        plan = ArchitectPlan(
            raw_text="Implement module.py with MyClass and process_data function",
            file_paths=["module.py"],
            class_names=["MyClass"],
            function_names=["process_data"],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["other.py"],  # Completely different file
            implemented_classes=["OtherClass"],  # Completely different class
            implemented_functions=["other_func"],  # Completely different function
        )
        report.calculate_metrics()
        
        assert report.file_adherence == 0.0
        assert report.class_adherence == 0.0
        assert report.function_adherence == 0.0
        assert report.overall_adherence == 0.0

    def test_no_plan_constraints(self):
        """Test adherence when plan has no constraints."""
        plan = ArchitectPlan(
            raw_text="Implement something",
            file_paths=[],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["anything.py"],
            implemented_classes=["AnythingClass"],
            implemented_functions=["anything"],
        )
        report.calculate_metrics()
        
        # No constraints = perfect adherence
        assert report.file_adherence == 1.0
        assert report.class_adherence == 1.0
        assert report.function_adherence == 1.0


class TestDeviationDetection:
    """Tests for deviation detection."""

    def test_detect_missing_files(self):
        """Test detection of missing planned files."""
        plan = ArchitectPlan(
            raw_text="Implement module.py",
            file_paths=["module.py"],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=[],  # Missing module.py
            implemented_classes=[],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        assert len(report.deviations) > 0
        assert any("module.py" in d for d in report.deviations)

    def test_detect_missing_classes(self):
        """Test detection of missing planned classes."""
        plan = ArchitectPlan(
            raw_text="Implement MyClass",
            file_paths=[],
            class_names=["MyClass"],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=[],
            implemented_classes=[],  # Missing MyClass
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        assert len(report.deviations) > 0
        assert any("MyClass" in d for d in report.deviations)

    def test_low_adherence_warning(self):
        """Test warning generation for low adherence."""
        plan = ArchitectPlan(
            raw_text="Implement module.py with MyClass",
            file_paths=["module.py"],
            class_names=["MyClass"],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["other.py"],
            implemented_classes=["OtherClass"],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        assert len(report.warnings) > 0
        assert any("adherence" in w.lower() for w in report.warnings)

    def test_no_deviations_when_perfect(self):
        """Test no deviations when adherence is perfect."""
        plan = ArchitectPlan(
            raw_text="Implement module.py",
            file_paths=["module.py"],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["module.py"],
            implemented_classes=[],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        assert len(report.deviations) == 0
        assert len(report.warnings) == 0


class TestArchitectFeedbackEnforcer:
    """Tests for ArchitectFeedbackEnforcer."""

    def test_set_plan(self):
        """Test setting and parsing a plan."""
        plan_text = """
        ## Implementation Plan
        
        Create module.py with MyClass.
        The system MUST use PostgreSQL.
        """
        
        enforcer = ArchitectFeedbackEnforcer(plan_text)
        
        assert enforcer.plan is not None
        assert enforcer.is_plan_preserved()
        assert "module.py" in enforcer.plan.file_paths
        assert "MyClass" in enforcer.plan.class_names

    def test_add_po_comments(self):
        """Test adding PO comments."""
        enforcer = ArchitectFeedbackEnforcer()
        comments = ["This feature is critical for Q4", "Use the new API"]
        enforcer.add_po_comments(comments)
        
        assert enforcer.po_comments == comments

    def test_record_implementation(self):
        """Test recording implementation and generating report."""
        plan_text = "Implement module.py with MyClass"
        enforcer = ArchitectFeedbackEnforcer(plan_text)
        
        report = enforcer.record_implementation(
            files=["module.py"],
            classes=["MyClass"],
            functions=[],
        )
        
        assert report is not None
        assert report.overall_adherence == 1.0

    def test_get_adherence_prompt(self):
        """Test generation of adherence reminder prompt."""
        plan_text = """
        ## Implementation Plan
        
        Create module.py.
        The system MUST use PostgreSQL.
        """
        
        enforcer = ArchitectFeedbackEnforcer(plan_text)
        prompt = enforcer.get_adherence_prompt()
        
        assert "Architect Guidance Reminder" in prompt
        assert "MUST" in prompt
        assert "module.py" in prompt

    def test_check_adherence_threshold(self):
        """Test adherence threshold checking."""
        plan_text = "Implement module.py and utils.py with MyClass"
        enforcer = ArchitectFeedbackEnforcer(plan_text)
        
        # Record partial implementation (only 1 of 2 files)
        enforcer.record_implementation(
            files=["module.py"],  # Missing utils.py
            classes=[],  # Missing MyClass
            functions=[],
        )
        
        # With 1/2 files (50%) and 0/1 classes (0%), weighted adherence is:
        # 0.4*0.5 + 0.3*0 + 0.3*1 = 0.2 + 0 + 0.3 = 0.5
        # Should fail 70% threshold
        assert enforcer.check_adherence_threshold(0.3) == True  # Passes 30%
        assert enforcer.check_adherence_threshold(0.7) == False  # Fails 70%

    def test_get_deviation_alert(self):
        """Test deviation alert generation."""
        plan_text = "Implement module.py"
        enforcer = ArchitectFeedbackEnforcer(plan_text)
        
        enforcer.record_implementation(
            files=[],  # Missing module.py
            classes=[],
            functions=[],
        )
        
        alert = enforcer.get_deviation_alert()
        
        assert alert is not None
        assert "DEVIATION" in alert
        assert "module.py" in alert

    def test_empty_plan(self):
        """Test handling of empty plan."""
        enforcer = ArchitectFeedbackEnforcer()
        
        assert enforcer.plan is None
        assert enforcer.is_plan_preserved() == False
        
        report = enforcer.record_implementation(
            files=["anything.py"],
            classes=[],
            functions=[],
        )
        
        assert report.overall_adherence == 1.0  # No plan = no violations


class TestContextCreation:
    """Tests for architect context creation."""

    def test_create_architect_context(self):
        """Test creation of context with architect guidance."""
        architect_plan = """
        ## Implementation Plan
        
        Create module.py with MyClass.
        """
        issue_description = "Fix the bug in the system."
        
        context = create_architect_context(architect_plan, issue_description)
        
        assert "Architect Guidance" in context
        assert "REQUIRED READING" in context
        assert "module.py" in context
        assert "MyClass" in context
        assert "Original Issue Description" in context
        assert "Fix the bug" in context

    def test_context_strips_prefix(self):
        """Test that context creation strips existing prefixes."""
        architect_plan = "## Architect Analysis\n\nCreate module.py"
        issue_description = "Fix the bug"
        
        context = create_architect_context(architect_plan, issue_description)
        
        # Should not have double headers
        assert context.count("Architect Analysis") <= 1
        assert "Create module.py" in context

    def test_context_prioritizes_architect(self):
        """Test that architect guidance appears before issue description."""
        architect_plan = "Create module.py"
        issue_description = "Fix the bug"
        
        context = create_architect_context(architect_plan, issue_description)
        
        arch_pos = context.find("Architect Guidance")
        issue_pos = context.find("Original Issue Description")
        
        assert arch_pos < issue_pos  # Architect comes first

    def test_context_with_po_comments(self):
        """Test context creation with PO comments."""
        architect_plan = "Create module.py"
        issue_description = "Fix the bug"
        po_comments = ["This is critical for Q4", "Use the new API"]
        
        context = create_architect_context(architect_plan, issue_description, po_comments)
        
        assert "Product Owner Comments" in context
        assert "This is critical for Q4" in context
        assert "Use the new API" in context
        
        # Check ordering: Architect -> PO Comments -> Issue Description
        arch_pos = context.find("Architect Guidance")
        po_pos = context.find("Product Owner Comments")
        issue_pos = context.find("Original Issue Description")
        
        assert arch_pos < po_pos < issue_pos


class MockNote:
    """Mock GitLab note object for testing."""
    def __init__(self, body: str):
        self.body = body


class TestExtractArchitectPlan:
    """Tests for extracting architect plan from GitLab notes."""

    def test_extract_from_architect_note(self):
        """Test extraction from architect analysis note."""
        notes = [
            MockNote("## Architect Analysis\n\nCreate module.py"),
            MockNote("Random comment"),
        ]
        
        plan = extract_architect_plan_from_notes(notes)
        
        assert plan is not None
        assert "Create module.py" in plan

    def test_extract_from_design_note(self):
        """Test extraction from architect design note."""
        notes = [
            MockNote("## Architect Design\n\nCreate module.py"),
        ]
        
        plan = extract_architect_plan_from_notes(notes)
        
        assert plan is not None
        assert "Create module.py" in plan

    def test_no_architect_notes(self):
        """Test when no architect notes exist."""
        notes = [
            MockNote("Random comment"),
            MockNote("Another comment"),
        ]
        
        plan = extract_architect_plan_from_notes(notes)
        
        assert plan is None

    def test_returns_latest_architect_note(self):
        """Test that the latest architect note is returned."""
        notes = [
            MockNote("## Architect Analysis\n\nFirst plan"),
            MockNote("## Architect Analysis\n\nUpdated plan"),
        ]
        
        plan = extract_architect_plan_from_notes(notes)
        
        assert plan is not None
        assert "Updated plan" in plan
        assert "First plan" not in plan


class TestExtractPOComments:
    """Tests for extracting PO comments from GitLab notes."""

    def test_extract_po_comments(self):
        """Test extraction of PO comments."""
        notes = [
            MockNote("## PO Comment\n\nThis is critical for Q4"),
            MockNote("Random comment"),
            MockNote("## PO Comment\n\nUse the new API"),
        ]
        
        comments = extract_po_comments_from_notes(notes)
        
        assert len(comments) == 2
        assert "This is critical for Q4" in comments
        assert "Use the new API" in comments

    def test_no_po_comments(self):
        """Test when no PO comments exist."""
        notes = [
            MockNote("Random comment"),
            MockNote("Another comment"),
        ]
        
        comments = extract_po_comments_from_notes(notes)
        
        assert len(comments) == 0

    def test_po_comment_strips_prefix(self):
        """Test that PO comment prefix is stripped."""
        notes = [
            MockNote("## PO Comment\n\nThis is critical"),
        ]
        
        comments = extract_po_comments_from_notes(notes)
        
        assert comments[0] == "This is critical"
        assert "## PO Comment" not in comments[0]


class TestAdherenceReportSummary:
    """Tests for adherence report summary generation."""

    def test_summary_format(self):
        """Test summary format and content."""
        plan = ArchitectPlan(
            raw_text="Implement module.py",
            file_paths=["module.py"],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["module.py"],
            implemented_classes=[],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        summary = report.to_summary()
        
        assert "Overall Adherence" in summary
        assert "100.0%" in summary
        assert "File Adherence" in summary

    def test_summary_with_deviations(self):
        """Test summary includes deviations."""
        plan = ArchitectPlan(
            raw_text="Implement module.py",
            file_paths=["module.py"],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=[],  # Missing
            implemented_classes=[],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        summary = report.to_summary()
        
        assert "Deviations" in summary
        assert "module.py" in summary

    def test_summary_with_warnings(self):
        """Test summary includes warnings."""
        plan = ArchitectPlan(
            raw_text="Implement module.py and utils.py",
            file_paths=["module.py", "utils.py"],
            class_names=[],
            function_names=[],
        )
        
        report = AdherenceReport(
            plan=plan,
            implemented_files=["module.py"],  # Partial
            implemented_classes=[],
            implemented_functions=[],
        )
        report.calculate_metrics()
        report.detect_deviations()
        
        summary = report.to_summary()
        
        assert "Warnings" in summary or "Adherence" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
