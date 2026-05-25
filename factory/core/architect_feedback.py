# Copyright 2024 SoftwareTeamFabrik contributors
# SPDX-License-Identifier: MIT

"""
Architect-Developer Feedback Loop Enforcement

This module enforces adherence to architectural guidance during implementation.
It addresses:
1. Feedback truncation: Ensures architect plans are preserved in context
2. Execution drift: Detects when developer deviates from architectural guidance
3. State loss: Maintains persistent record of architectural decisions
4. PO comment integration: Incorporates Product Owner comments into architectural analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from factory.core.constants import ARCH_NOTE_PREFIX


@dataclass
class ArchitectPlan:
    """Represents an architect's implementation plan."""
    raw_text: str
    file_paths: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    function_names: list[str] = field(default_factory=list)
    key_decisions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    
    def extract_structure(self) -> None:
        """Extract structural elements from the plan text."""
        # Extract file paths (e.g., factory/core/module.py)
        # Match full paths like: path/to/file.py, module.py, etc.
        file_pattern = r'([a-zA-Z_][\w\-/]*[a-zA-Z_][\w\-]*\.(?:py|yml|yaml|md|txt|json|js|ts|css|html)|[a-zA-Z_][\w\-]*\.(?:py|yml|yaml|md|txt|json|js|ts|css|html))'
        self.file_paths = list(set(re.findall(file_pattern, self.raw_text)))
        
        # Extract class names (e.g., class ClassName or mentions of PascalCase names)
        # First try explicit "class ClassName" pattern
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9]*)'
        class_names = list(set(re.findall(class_pattern, self.raw_text)))
        
        # Also look for PascalCase names that might be class references
        # but exclude common words
        pascal_pattern = r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b'
        pascal_names = set(re.findall(pascal_pattern, self.raw_text))
        
        # Combine and deduplicate
        self.class_names = list(set(class_names) | pascal_names)
        
        # Extract function names (e.g., def function_name)
        func_pattern = r'def\s+([a-z_][a-zA-Z0-9_]*)'
        self.function_names = list(set(re.findall(func_pattern, self.raw_text)))
        
        # Extract key decisions (lines with "must", "should", "will")
        decision_keywords = ['must', 'should', 'will', 'required', 'mandatory']
        for line in self.raw_text.split('\n'):
            if any(kw in line.lower() for kw in decision_keywords):
                self.key_decisions.append(line.strip())


@dataclass
class AdherenceReport:
    """Report on developer adherence to architect guidance."""
    plan: ArchitectPlan
    implemented_files: list[str] = field(default_factory=list)
    implemented_classes: list[str] = field(default_factory=list)
    implemented_functions: list[str] = field(default_factory=list)
    
    # Adherence metrics
    file_adherence: float = 0.0  # 0.0 to 1.0
    class_adherence: float = 0.0
    function_adherence: float = 0.0
    overall_adherence: float = 0.0
    
    # Deviations detected
    deviations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def calculate_metrics(self) -> None:
        """Calculate adherence metrics based on implementation vs plan."""
        # File adherence - use basename comparison for flexibility
        if self.plan.file_paths:
            plan_basenames = {Path(f).name for f in self.plan.file_paths}
            impl_basenames = {Path(f).name for f in self.implemented_files}
            matched_files = plan_basenames & impl_basenames
            self.file_adherence = len(matched_files) / len(plan_basenames) if plan_basenames else 1.0
        else:
            self.file_adherence = 1.0  # No files specified = no constraint
        
        # Class adherence
        if self.plan.class_names:
            matched_classes = set(self.implemented_classes) & set(self.plan.class_names)
            self.class_adherence = len(matched_classes) / len(self.plan.class_names)
        else:
            self.class_adherence = 1.0
        
        # Function adherence
        if self.plan.function_names:
            matched_funcs = set(self.implemented_functions) & set(self.plan.function_names)
            self.function_adherence = len(matched_funcs) / len(self.plan.function_names)
        else:
            self.function_adherence = 1.0
        
        # Overall adherence (weighted average)
        weights = {'file': 0.4, 'class': 0.3, 'function': 0.3}
        self.overall_adherence = (
            weights['file'] * self.file_adherence +
            weights['class'] * self.class_adherence +
            weights['function'] * self.function_adherence
        )
    
    def detect_deviations(self) -> None:
        """Detect significant deviations from the plan."""
        # Check for missing required files (using basename comparison)
        plan_basenames = {Path(f).name: f for f in self.plan.file_paths}
        impl_basenames = {Path(f).name for f in self.implemented_files}
        
        for basename, full_path in plan_basenames.items():
            if basename not in impl_basenames:
                self.deviations.append(f"Planned file not implemented: {full_path}")
        
        # Check for missing required classes
        for planned_class in self.plan.class_names:
            if planned_class not in self.implemented_classes:
                self.deviations.append(f"Planned class not implemented: {planned_class}")
        
        # Check for missing required functions
        for planned_func in self.plan.function_names:
            if planned_func not in self.implemented_functions:
                self.deviations.append(f"Planned function not implemented: {planned_func}")
        
        # Check adherence thresholds
        if self.overall_adherence < 0.5:
            self.warnings.append(
                f"Low overall adherence ({self.overall_adherence:.1%}) - "
                "implementation may not match architectural intent"
            )
        
        if self.file_adherence < 0.5:
            self.warnings.append(
                f"Low file adherence ({self.file_adherence:.1%}) - "
                "consider implementing planned file structure"
            )
    
    def to_summary(self) -> str:
        """Generate a human-readable summary of adherence."""
        lines = [
            "## Architect Adherence Report",
            "",
            f"**Overall Adherence**: {self.overall_adherence:.1%}",
            "",
            "### Metrics",
            f"- File Adherence: {self.file_adherence:.1%}",
            f"- Class Adherence: {self.class_adherence:.1%}",
            f"- Function Adherence: {self.function_adherence:.1%}",
        ]
        
        if self.deviations:
            lines.extend([
                "",
                "### Deviations Detected",
            ])
            for dev in self.deviations:
                lines.append(f"- ⚠️ {dev}")
        
        if self.warnings:
            lines.extend([
                "",
                "### Warnings",
            ])
            for warn in self.warnings:
                lines.append(f"- ⚠️ {warn}")
        
        if not self.deviations and not self.warnings:
            lines.extend([
                "",
                "✅ Implementation follows architectural guidance.",
            ])
        
        return "\n".join(lines)


class ArchitectFeedbackEnforcer:
    """
    Enforces architect-developer feedback loop during implementation.
    
    This class:
    1. Parses architect plans to extract structural requirements
    2. Monitors implementation progress against the plan
    3. Detects and reports deviations from architectural guidance
    4. Ensures architect feedback is preserved in context
    5. Incorporates PO comments into architectural analysis
    """

    ARCH_DESIGN_PREFIX = "## Architect Design\n\n"
    PO_COMMENT_PREFIX = "## PO Comment\n\n"
    
    def __init__(self, plan_text: str = ""):
        """Initialize with architect plan text."""
        self.plan: Optional[ArchitectPlan] = None
        self.report: Optional[AdherenceReport] = None
        self._plan_preserved = False
        self.po_comments: List[str] = []
        
        if plan_text:
            self.set_plan(plan_text)
    
    def set_plan(self, plan_text: str) -> None:
        """Set and parse the architect plan."""
        # Strip common prefixes if present
        clean_text = plan_text
        for prefix in [ARCH_NOTE_PREFIX, self.ARCH_DESIGN_PREFIX]:
            if clean_text.startswith(prefix):
                clean_text = clean_text[len(prefix):]
                break
        
        self.plan = ArchitectPlan(raw_text=clean_text)
        self.plan.extract_structure()
        self._plan_preserved = True
    
    def add_po_comments(self, comments: List[str]) -> None:
        """Add Product Owner comments to be considered in architectural analysis."""
        self.po_comments = comments
    
    def is_plan_preserved(self) -> bool:
        """Check if architect plan is preserved in context."""
        return self._plan_preserved
    
    def record_implementation(
        self,
        files: list[str],
        classes: Optional[list[str]] = None,
        functions: Optional[list[str]] = None,
    ) -> AdherenceReport:
        """
        Record implemented elements and generate adherence report.
        
        Args:
            files: List of file paths implemented
            classes: List of class names implemented (optional)
            functions: List of function names implemented (optional)
        
        Returns:
            AdherenceReport with metrics and deviations
        """
        if not self.plan:
            # No plan to compare against - create empty report
            report = AdherenceReport(
                plan=ArchitectPlan(raw_text=""),
                implemented_files=files,
                implemented_classes=classes or [],
                implemented_functions=functions or [],
            )
            report.file_adherence = 1.0
            report.class_adherence = 1.0
            report.function_adherence = 1.0
            report.overall_adherence = 1.0
            self.report = report
            return report
        
        report = AdherenceReport(
            plan=self.plan,
            implemented_files=files,
            implemented_classes=classes or [],
            implemented_functions=functions or [],
        )
        report.calculate_metrics()
        report.detect_deviations()
        self.report = report
        return report
    
    def get_adherence_prompt(self) -> str:
        """
        Generate a prompt reminder for the developer to follow architect guidance.
        
        This should be included in the context when the developer is working.
        """
        if not self.plan:
            return ""
        
        prompt_parts = [
            "",
            "## ⚠️ Architect Guidance Reminder",
            "",
            "The Architect has provided a detailed implementation plan. "
            "You MUST follow this plan unless you encounter a technical blocker. "
            "If you deviate from the plan, you MUST explain why in your commit message.",
            "",
            "### Key Requirements from Architect:",
        ]
        
        # Add key decisions
        for i, decision in enumerate(self.plan.key_decisions[:5], 1):
            prompt_parts.append(f"{i}. {decision}")
        
        if self.plan.file_paths:
            prompt_parts.append("")
            prompt_parts.append("### Required Files:")
            for f in self.plan.file_paths:
                prompt_parts.append(f"- `{f}`")
        
        if self.plan.class_names:
            prompt_parts.append("")
            prompt_parts.append("### Required Classes:")
            for c in self.plan.class_names:
                prompt_parts.append(f"- `{c}`")
        
        prompt_parts.append("")
        prompt_parts.append("---")
        
        return "\n".join(prompt_parts)
    
    def check_adherence_threshold(self, threshold: float = 0.7) -> bool:
        """
        Check if current adherence meets the threshold.
        
        Args:
            threshold: Minimum acceptable adherence (0.0 to 1.0)
        
        Returns:
            True if adherence meets threshold, False otherwise
        """
        if not self.report:
            return True  # No report yet = no violations
        
        return self.report.overall_adherence >= threshold
    
    def get_deviation_alert(self) -> Optional[str]:
        """
        Get an alert message if deviations are detected.
        
        Returns:
            Alert message string if deviations exist, None otherwise
        """
        if not self.report or not self.report.deviations:
            return None
        
        alert_lines = [
            "🚨 ARCHITECT DEVIATION DETECTED",
            "",
            "The following deviations from the architectural plan were detected:",
        ]
        
        for dev in self.report.deviations:
            alert_lines.append(f"- {dev}")
        
        alert_lines.append("")
        alert_lines.append(
            "Please review the architect's plan and adjust your implementation, "
            "or document why the deviation is necessary."
        )
        
        return "\n".join(alert_lines)


def extract_architect_plan_from_notes(notes: list) -> Optional[str]:
    """
    Extract the latest architect plan from GitLab issue notes.
    
    Args:
        notes: List of GitLab note objects
    
    Returns:
        Architect plan text if found, None otherwise
    """
    arch_notes = []
    
    for note in notes:
        body = getattr(note, 'body', '') or ''
        if body.startswith("## Architect Analysis\n\n") or \
           body.startswith("## Architect Design\n\n"):
            arch_notes.append(body)
    
    if not arch_notes:
        return None
    
    # Return the latest architect note
    return arch_notes[-1]


def extract_po_comments_from_notes(notes: list) -> List[str]:
    """
    Extract Product Owner comments from GitLab issue notes.
    
    Args:
        notes: List of GitLab note objects
    
    Returns:
        List of PO comment texts
    """
    po_comments = []
    
    for note in notes:
        body = getattr(note, 'body', '') or ''
        if body.startswith("## PO Comment\n\n"):
            # Strip the prefix to get the actual comment
            comment_text = body[len("## PO Comment\n\n"):]
            po_comments.append(comment_text)
    
    return po_comments


def create_architect_context(
    architect_plan: str,
    issue_description: str,
    po_comments: Optional[List[str]] = None,
) -> str:
    """
    Create the full context for developer with architect guidance and PO comments.
    
    This ensures architect feedback is not truncated and is prominently
    displayed in the developer's context. PO comments are also included
    to provide additional business context.
    
    Args:
        architect_plan: The architect's analysis and plan
        issue_description: The original issue description
        po_comments: List of Product Owner comments (optional)
    
    Returns:
        Combined context string with architect guidance first, then PO comments, then issue description
    """
    # Strip any existing prefixes from the plan
    clean_plan = architect_plan
    for prefix in ["## Architect Analysis\n\n", "## Architect Design\n\n"]:
        if clean_plan.startswith(prefix):
            clean_plan = clean_plan[len(prefix):]
            break
    
    # Build context with architect guidance first
    context_parts = [
        "## 🏗️ Architect Guidance (REQUIRED READING)",
        "",
        "The following plan was provided by the Architect agent. "
        "You MUST follow this plan unless you encounter a technical blocker.",
        "If you must deviate, explain why in your commit message.",
        "",
        clean_plan,
        "",
        "---",
    ]
    
    # Add PO comments if available
    if po_comments:
        context_parts.extend([
            "",
            "## 💬 Product Owner Comments",
            "",
            "The following comments were provided by the Product Owner:",
        ])
        
        for i, comment in enumerate(po_comments, 1):
            context_parts.extend([
                f"",
                f"### PO Comment {i}:",
                "",
                comment.strip(),
            ])
        
        context_parts.extend([
            "",
            "---",
        ])
    
    # Add original issue description
    context_parts.extend([
        "",
        "## Original Issue Description",
        "",
        issue_description or "(no description provided)",
    ])
    
    return "\n".join(context_parts)
