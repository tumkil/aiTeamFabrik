import pytest
from dotenv import load_dotenv

load_dotenv()  # makes GITLAB_TOKEN available to integration tests

# Define pytest markers for test categorization
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as requiring GitLab API access"
    )
