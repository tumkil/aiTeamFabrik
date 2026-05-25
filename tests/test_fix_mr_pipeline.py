"""Test cases for the fix_mr_pipeline command."""

import pytest
from unittest.mock import patch, MagicMock
from factory.commands.fix_mr_pipeline import app


def test_fix_mr_pipeline_help():
    """Test the fix_mr_pipeline command help message."""
    from typer.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Act on failed pipelines for a merge request" in result.output


def test_fix_mr_pipeline_retry_success():
    """Test the fix_mr_pipeline command with successful retry."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.return_value = MagicMock(
            iid=1,
            title="Test MR",
            pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                status="failed",
                retry=MagicMock()
            )]))
        )
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1", "--retry"])
        assert result.exit_code == 0
        assert "Pipeline retry triggered" in result.output


def test_fix_mr_pipeline_repair_success():
    """Test the fix_mr_pipeline command with successful repair."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        with patch("factory.commands.fix_mr_pipeline.CompetenceManager") as mock_cm_class:
            with patch("factory.commands.refine.run_mr_refine") as mock_refine:
                mock_gl_client = MagicMock()
                mock_gl_client.connect.return_value = (True, "Connected")
                mock_gl_client.project.mergerequests.get.return_value = MagicMock(
                    iid=1,
                    title="Test MR",
                    pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                        status="failed"
                    )]))
                )
                mock_gl_client_class.return_value = mock_gl_client
                
                mock_cm = MagicMock()
                mock_cm.get.return_value = MagicMock(name="developer")
                mock_cm_class.return_value = mock_cm
                
                mock_refine.return_value = MagicMock(
                    commit_sha="abc123456789",
                    needs_continuation=False
                )
                
                from typer.testing import CliRunner
                
                runner = CliRunner()
                result = runner.invoke(app, ["--mr-iid", "1"])
                assert result.exit_code == 0
                assert "Repair successful" in result.output


def test_fix_mr_pipeline_no_pipelines():
    """Test the fix_mr_pipeline command when no pipelines are found."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.return_value = MagicMock(
            iid=1,
            title="Test MR",
            pipelines=MagicMock(list=MagicMock(return_value=[]))
        )
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1"])
        assert result.exit_code == 0
        assert "No pipelines found" in result.output


def test_fix_mr_pipeline_connection_failure():
    """Test the fix_mr_pipeline command when GitLab connection fails."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (False, "Connection failed")
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1"])
        assert result.exit_code == 1
        assert "GitLab connection failed" in result.output


def test_fix_mr_pipeline_get_mr_failure():
    """Test the fix_mr_pipeline command when getting the MR fails."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.side_effect = Exception("MR not found")
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1"])
        assert result.exit_code == 1
        assert "Failed to get MR" in result.output


def test_fix_mr_pipeline_retry_failure():
    """Test the fix_mr_pipeline command when retrying the pipeline fails."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.return_value = MagicMock(
            iid=1,
            title="Test MR",
            pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                status="failed",
                retry=MagicMock(side_effect=Exception("Retry failed"))
            )]))
        )
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1", "--retry"])
        assert result.exit_code == 1
        assert "Failed to retry pipeline" in result.output


def test_fix_mr_pipeline_repair_failure():
    """Test the fix_mr_pipeline command when repair fails."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        with patch("factory.commands.fix_mr_pipeline.CompetenceManager") as mock_cm_class:
            with patch("factory.commands.fix_mr_pipeline.fix_mr_pipeline") as mock_fix:
                mock_gl_client = MagicMock()
                mock_gl_client.connect.return_value = (True, "Connected")
                mock_gl_client.project.mergerequests.get.return_value = MagicMock(
                    iid=1,
                    title="Test MR",
                    pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                        status="failed"
                    )]))
                )
                mock_gl_client_class.return_value = mock_gl_client
                
                mock_cm = MagicMock()
                mock_cm.get.return_value = MagicMock(name="developer")
                mock_cm_class.return_value = mock_cm
                
                mock_fix.return_value = False
                
                from typer.testing import CliRunner
                
                runner = CliRunner()
                result = runner.invoke(app, ["--mr-iid", "1"], catch_exceptions=False)
                assert result.exit_code == 1


def test_fix_mr_pipeline_running_pipeline():
    """Test the fix_mr_pipeline command when pipeline is running."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.return_value = MagicMock(
            iid=1,
            title="Test MR",
            pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                status="running"
            )]))
        )
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1"])
        assert result.exit_code == 0
        assert "still running" in result.output


def test_fix_mr_pipeline_success_pipeline():
    """Test the fix_mr_pipeline command when pipeline is successful."""
    with patch("factory.commands.fix_mr_pipeline.GitLabClient") as mock_gl_client_class:
        mock_gl_client = MagicMock()
        mock_gl_client.connect.return_value = (True, "Connected")
        mock_gl_client.project.mergerequests.get.return_value = MagicMock(
            iid=1,
            title="Test MR",
            pipelines=MagicMock(list=MagicMock(return_value=[MagicMock(
                status="success"
            )]))
        )
        mock_gl_client_class.return_value = mock_gl_client
        
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ["--mr-iid", "1"])
        assert result.exit_code == 0
        assert "Pipeline passed" in result.output
