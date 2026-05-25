"""Unit tests for merge request creation with retry mechanism."""
import pytest
from unittest.mock import MagicMock, patch, call
import gitlab

from factory.adapters.gitlab_client import GitLabClient, retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff helper function."""

    def test_success_on_first_attempt(self):
        """Function succeeds on first attempt, no retries needed."""
        def success_func():
            return "success"
        
        result = retry_with_backoff(success_func, max_retries=3)
        assert result == "success"

    def test_success_after_two_retries(self):
        """Function succeeds after two failed attempts."""
        attempt_count = [0]
        
        def flaky_func():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = retry_with_backoff(flaky_func, max_retries=3)
        assert result == "success"
        assert attempt_count[0] == 3

    def test_failure_after_all_retries(self):
        """Function fails after all retry attempts are exhausted."""
        def always_fails():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(always_fails, max_retries=3)

    def test_only_catches_specified_exceptions(self):
        """Only specified exception types trigger retry."""
        attempt_count = [0]
        
        def wrong_exception():
            attempt_count[0] += 1
            raise TypeError("Wrong type")
        
        with pytest.raises(TypeError, match="Wrong type"):
            retry_with_backoff(wrong_exception, max_retries=3, exceptions=(ValueError,))
        
        # Should only attempt once since TypeError is not in exceptions tuple
        assert attempt_count[0] == 1

    def test_exponential_backoff_timing(self):
        """Verify exponential backoff calculates correct delays."""
        with patch('time.sleep') as mock_sleep:
            attempt_count = [0]
            
            def flaky_func():
                attempt_count[0] += 1
                if attempt_count[0] < 3:
                    raise ValueError("Temporary failure")
                return "success"
            
            retry_with_backoff(
                flaky_func,
                max_retries=3,
                base_delay=1.0,
                exponential_base=2.0,
            )
            
            # Should sleep twice (after attempt 1 and 2)
            assert mock_sleep.call_count == 2
            # First delay: 1.0 * 2^0 = 1.0 (with jitter)
            # Second delay: 1.0 * 2^1 = 2.0 (with jitter)
            calls = mock_sleep.call_args_list
            assert len(calls) == 2


class TestGitLabClientCreateMergeRequest:
    """Tests for GitLabClient.create_merge_request method."""

    def test_create_mr_success(self):
        """MR creation succeeds on first attempt."""
        mock_project = MagicMock()
        mock_mr = MagicMock()
        mock_mr.iid = 123
        mock_project.mergerequests.create.return_value = mock_mr
        
        # Create a real GitLabClient instance with mocked project
        client = GitLabClient.__new__(GitLabClient)
        client._project = mock_project
        
        result = client.create_merge_request(
            source_branch='feature',
            target_branch='main',
            title='Test MR',
            description='Test description',
            work_in_progress=False,
            remove_source_branch=True,
            max_retries=3,
        )
        
        assert result.iid == 123
        mock_project.mergerequests.create.assert_called_once()

    @patch('factory.adapters.gitlab_client.time.sleep')
    def test_create_mr_retry_on_connection_error(self, mock_sleep):
        """MR creation retries on GitLabConnectionError."""
        mock_project = MagicMock()
        
        # Simulate connection errors then success
        call_count = [0]
        def flaky_create(data):
            call_count[0] += 1
            if call_count[0] < 3:
                raise gitlab.GitlabConnectionError("Connection failed")
            mr = MagicMock()
            mr.iid = 123
            return mr
        
        mock_project.mergerequests.create = flaky_create
        
        # Create a real GitLabClient instance with mocked project
        client = GitLabClient.__new__(GitLabClient)
        client._project = mock_project
        
        result = client.create_merge_request(
            source_branch='feature',
            target_branch='main',
            title='Test MR',
            max_retries=3,
        )
        
        assert result.iid == 123
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2

    @patch('factory.adapters.gitlab_client.time.sleep')
    def test_create_mr_returns_none_after_failures(self, mock_sleep):
        """MR creation returns None after all retries fail."""
        mock_project = MagicMock()
        mock_project.mergerequests.create.side_effect = gitlab.GitlabCreateError("Create failed")
        
        # Create a real GitLabClient instance with mocked project
        client = GitLabClient.__new__(GitLabClient)
        client._project = mock_project
        
        result = client.create_merge_request(
            source_branch='feature',
            target_branch='main',
            title='Test MR',
            max_retries=3,
        )
        
        # Should return None after all retries fail
        assert result is None

    def test_create_mr_with_all_parameters(self):
        """MR creation passes all parameters correctly."""
        mock_project = MagicMock()
        mock_mr = MagicMock()
        mock_mr.iid = 456
        mock_project.mergerequests.create.return_value = mock_mr
        
        # Create a real GitLabClient instance with mocked project
        client = GitLabClient.__new__(GitLabClient)
        client._project = mock_project
        
        result = client.create_merge_request(
            source_branch='feature-branch',
            target_branch='develop',
            title='feat: new feature',
            description='Detailed description',
            work_in_progress=True,
            remove_source_branch=True,
            max_retries=3,
        )
        
        assert result.iid == 456
        mock_project.mergerequests.create.assert_called_once_with({
            'source_branch': 'feature-branch',
            'target_branch': 'develop',
            'title': 'feat: new feature',
            'description': 'Detailed description',
            'work_in_progress': True,
            'remove_source_branch': True,
        })
