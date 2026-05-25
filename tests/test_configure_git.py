"""Tests for _configure_git() and _git_config_is_set() in factory.commands.run.

These tests verify that:
1. Git user.name and user.email are set when not already configured.
2. Existing Git identity settings are NOT overwritten.
3. Partially configured identities (only name or only email) are handled correctly.
4. Values are written with ``--local`` scope so they never leak to global config.
"""

import subprocess
from unittest.mock import patch, MagicMock, call

import pytest

from factory.commands.run import _configure_git, _git_config_is_set


# ---------------------------------------------------------------------------
# _git_config_is_set
# ---------------------------------------------------------------------------

class TestGitConfigIsSet:
    """Tests for the _git_config_is_set helper."""

    def test_returns_true_when_config_is_set(self):
        """If git config --get exits with 0, the key is considered set."""
        with patch("factory.commands.run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _git_config_is_set("user.name") is True
            mock_run.assert_called_once_with(
                ["git", "config", "--get", "user.name"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_returns_false_when_config_is_unset(self):
        """If git config --get exits with 1, the key is considered unset."""
        with patch("factory.commands.run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert _git_config_is_set("user.name") is False

    def test_returns_false_when_git_config_fails(self):
        """If git config --get exits with a non-zero code (e.g. 128 for bad repo), treat as unset."""
        with patch("factory.commands.run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=128)
            assert _git_config_is_set("user.email") is False

    def test_passes_correct_key(self):
        """The helper should pass the exact key string to git config --get."""
        with patch("factory.commands.run.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _git_config_is_set("user.email")
            mock_run.assert_called_once_with(
                ["git", "config", "--get", "user.email"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


# ---------------------------------------------------------------------------
# _configure_git
# ---------------------------------------------------------------------------

class TestConfigureGit:
    """Tests for the _configure_git function."""

    def test_sets_name_and_email_when_both_unset(self):
        """When neither user.name nor user.email is set, both should be configured."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            # Both configs are unset
            mock_is_set.return_value = False
            mock_run.return_value = MagicMock(returncode=0)

            _configure_git()

            # Should have checked both keys
            assert mock_is_set.call_count == 2
            mock_is_set.assert_any_call("user.name")
            mock_is_set.assert_any_call("user.email")

            # Should have set both values with --local
            assert mock_run.call_count == 2
            mock_run.assert_any_call(
                ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
                check=True,
            )
            mock_run.assert_any_call(
                ["git", "config", "--local", "user.email", "factory@example.com"],
                check=True,
            )

    def test_does_not_overwrite_existing_name(self):
        """If user.name is already set, it must NOT be overwritten."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            # user.name is set, user.email is not
            def is_set_side_effect(key):
                return key == "user.name"
            mock_is_set.side_effect = is_set_side_effect
            mock_run.return_value = MagicMock(returncode=0)

            _configure_git()

            # Only email should be set
            assert mock_run.call_count == 1
            mock_run.assert_called_once_with(
                ["git", "config", "--local", "user.email", "factory@example.com"],
                check=True,
            )

    def test_does_not_overwrite_existing_email(self):
        """If user.email is already set, it must NOT be overwritten."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            # user.name is not set, user.email is set
            def is_set_side_effect(key):
                return key == "user.email"
            mock_is_set.side_effect = is_set_side_effect
            mock_run.return_value = MagicMock(returncode=0)

            _configure_git()

            # Only name should be set
            assert mock_run.call_count == 1
            mock_run.assert_called_once_with(
                ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
                check=True,
            )

    def test_does_not_overwrite_existing_identity(self):
        """If both user.name and user.email are already set, neither should be overwritten."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            # Both configs are already set
            mock_is_set.return_value = True

            _configure_git()

            # Should have checked both keys but not called git config set
            assert mock_is_set.call_count == 2
            mock_run.assert_not_called()

    def test_sets_name_first_then_email(self):
        """Configuration order should be: user.name, then user.email."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            mock_is_set.return_value = False
            mock_run.return_value = MagicMock(returncode=0)

            _configure_git()

            calls = mock_run.call_args_list
            assert calls[0] == call(
                ["git", "config", "--local", "user.name", "SoftwareTeamFabrik"],
                check=True,
            )
            assert calls[1] == call(
                ["git", "config", "--local", "user.email", "factory@example.com"],
                check=True,
            )

    def test_uses_local_scope(self):
        """Values must be written with --local to avoid polluting global config."""
        with patch("factory.commands.run._git_config_is_set") as mock_is_set, \
             patch("factory.commands.run.subprocess.run") as mock_run:
            mock_is_set.return_value = False
            mock_run.return_value = MagicMock(returncode=0)

            _configure_git()

            for call_args in mock_run.call_args_list:
                cmd = call_args.args[0]
                assert "--local" in cmd, (
                    f"git config command {cmd} is missing --local flag"
                )