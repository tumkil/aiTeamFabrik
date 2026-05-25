"""Unit tests for LlmRouter.anthropic_complete — runs fully offline via stubbing."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import requests
from factory.adapters.llm_router import LlmRouter, LLMTimeoutError, LLMTransientError, LLMPermanentError
from factory.core.competence import AgentProfile


def _anthropic_agent(tools=None, temperature=None) -> AgentProfile:
    return AgentProfile(
        name="anthropic-agent",
        display_name="Anthropic Agent",
        model="claude-3-5-sonnet-20240620",
        provider="anthropic",
        tools=tools or [],
        temperature=temperature,
    )


# Mock anthropic module for all tests
@pytest.fixture(autouse=True)
def mock_anthropic_module():
    """Fixture to mock the anthropic module for all tests."""
    anthropic_mock = MagicMock()
    # Create a mock Anthropic class
    mock_anthropic_class = MagicMock()
    anthropic_mock.Anthropic = mock_anthropic_class
    
    # Add the mock to sys.modules so imports work
    with patch.dict('sys.modules', {'anthropic': anthropic_mock}):
        yield anthropic_mock


def test_anthropic_complete_missing_key_raises():
    """anthropic_complete should raise RuntimeError when ANTHROPIC_API_KEY is missing."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
        router = LlmRouter()
        agent = _anthropic_agent()
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            router.anthropic_complete(agent, system="test", messages=[{"role": "user", "content": "hello"}])


def test_anthropic_complete_basic_call(mock_anthropic_module):
    """Test basic anthropic_complete call with mocked anthropic client."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response text")]
    mock_message.usage = Mock(input_tokens=10, output_tokens=20)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent, 
            system="test system",
            messages=[{"role": "user", "content": "test prompt"}]
        )
        
        # Verify the client was called correctly
        assert mock_client.messages.create.called
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20240620"
        assert call_kwargs["max_tokens"] == 8192
        assert len(call_kwargs["system"]) == 1
        assert call_kwargs["system"][0]["text"] == "test system"
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]
        
        # Result should be the raw message object
        assert result == mock_message


def test_anthropic_complete_with_tools(mock_anthropic_module):
    """Test anthropic_complete with tools parameter."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response with tools")]
    mock_message.usage = Mock(input_tokens=15, output_tokens=25)
    
    web_search_tool = {
        "type": "web_search_20250305",
        "name": "web_search",
    }
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent(tools=["web_search"])
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "search something"}],
            tools=[web_search_tool]
        )
        
        # Verify tools were passed to the API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == [web_search_tool]
        assert result == mock_message


def test_anthropic_complete_with_temperature(mock_anthropic_module):
    """Test anthropic_complete with temperature parameter."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=5, output_tokens=10)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent(temperature=0.7)
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "test"}]
        )
        
        # Verify temperature was passed and clamped if > 1.0
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert result == mock_message


def test_anthropic_complete_temperature_above_one_clamped(mock_anthropic_module):
    """Test that temperature > 1.0 is clamped to 1.0 for Anthropic."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=5, output_tokens=10)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent(temperature=1.5)  # Above max
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "test"}]
        )
        
        # Verify temperature was clamped to 1.0
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 1.0
        assert result == mock_message


def test_anthropic_complete_with_conversation_history(mock_anthropic_module):
    """Test anthropic_complete with multiple messages (conversation history)."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=20, output_tokens=15)
    
    messages = [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "first response"},
        {"role": "user", "content": "follow-up"},
    ]
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=messages
        )
        
        # Verify all messages were passed
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == messages
        assert result == mock_message


def test_anthropic_complete_custom_max_tokens(mock_anthropic_module):
    """Test anthropic_complete with custom max_tokens parameter."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=10, output_tokens=20)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=4096
        )
        
        # Verify custom max_tokens was used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096
        assert result == mock_message


def test_anthropic_complete_retry_on_timeout(mock_anthropic_module):
    """Test that anthropic_complete retries on timeout errors."""
    mock_client = Mock()
    
    # First two calls raise timeout, third succeeds
    mock_client.messages.create.side_effect = [
        requests.exceptions.Timeout("first timeout"),
        requests.exceptions.Timeout("second timeout"),
        Mock(content=[Mock(text="success")], usage=Mock(input_tokens=10, output_tokens=20))
    ]
    
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent,
            system="test",
            messages=[{"role": "user", "content": "test"}]
        )
        
        # Should have been called 3 times
        assert mock_client.messages.create.call_count == 3
        assert result.content[0].text == "success"


def test_anthropic_complete_retry_exhausted_raises_timeout(mock_anthropic_module):
    """Test that anthropic_complete raises LLMTimeoutError when retries exhausted."""
    mock_client = Mock()
    mock_client.messages.create.side_effect = requests.exceptions.Timeout("timeout")
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        with pytest.raises(LLMTimeoutError, match="Anthropic API timeout"):
            router.anthropic_complete(
                agent,
                system="test",
                messages=[{"role": "user", "content": "test"}]
            )


def test_anthropic_complete_retry_on_transient_error(mock_anthropic_module):
    """Test that anthropic_complete retries on transient errors."""
    mock_client = Mock()
    
    # First call raises transient error, second succeeds
    mock_client.messages.create.side_effect = [
        requests.exceptions.ConnectionError("connection error"),
        Mock(content=[Mock(text="success")], usage=Mock(input_tokens=10, output_tokens=20))
    ]
    
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent,
            system="test",
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert mock_client.messages.create.call_count == 2
        assert result.content[0].text == "success"


def test_anthropic_complete_permanent_error_no_retry(mock_anthropic_module):
    """Test that anthropic_complete doesn't retry on permanent errors."""
    mock_client = Mock()
    
    # Raise a permanent error (e.g., invalid API key)
    mock_client.messages.create.side_effect = RuntimeError("Invalid API key")
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        with pytest.raises(LLMPermanentError, match="Anthropic API permanent error"):
            router.anthropic_complete(
                agent,
                system="test",
                messages=[{"role": "user", "content": "test"}]
            )
        
        # Should only be called once (no retry)
        assert mock_client.messages.create.call_count == 1


def test_anthropic_complete_budget_check(mock_anthropic_module):
    """Test that anthropic_complete performs budget check when budget manager exists."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=10, output_tokens=20)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        # Create router with budget manager
        router = LlmRouter()
        agent = _anthropic_agent()
        
        # Mock the budget manager's can_consume method
        if router._budget_manager:
            with patch.object(router._budget_manager, 'can_consume') as mock_can_consume:
                mock_can_consume.return_value = (True, "")  # allowed, no warning
                
                result = router.anthropic_complete(
                    agent,
                    system="test system",
                    messages=[{"role": "user", "content": "test prompt"}]
                )
                
                # Verify budget check was called
                assert mock_can_consume.called
                assert result == mock_message
        else:
            # If no budget manager, just verify it works without error
            result = router.anthropic_complete(
                agent,
                system="test system",
                messages=[{"role": "user", "content": "test prompt"}]
            )
            assert result == mock_message


def test_anthropic_complete_model_resolution(mock_anthropic_module):
    """Test that anthropic_complete uses agent.resolve_model() for model selection."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=10, output_tokens=20)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        
        # Create agent with model dict (provider-specific models)
        agent = AgentProfile(
            name="multi-model-agent",
            display_name="Multi-Model Agent",
            model={
                "anthropic": "claude-3-opus-20240229",
                "mistral": "mistral-large-latest"
            },
            provider="anthropic",
        )
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "test"}]
        )
        
        # Verify the resolved anthropic model was used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"
        assert result == mock_message


def test_anthropic_complete_empty_messages(mock_anthropic_module):
    """Test anthropic_complete with empty messages list."""
    mock_message = Mock()
    mock_message.content = [Mock(text="response")]
    mock_message.usage = Mock(input_tokens=5, output_tokens=10)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent()
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[]
        )
        
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == []
        assert result == mock_message


def test_anthropic_complete_complex_content_blocks(mock_anthropic_module):
    """Test anthropic_complete handles complex content blocks (text + tool_use)."""
    # Create a mock message with mixed content blocks
    mock_text_block = Mock(text="Here's some information")
    mock_tool_use_block = Mock(type="tool_use", id="tool123", name="web_search", input={"query": "test"})
    mock_message = Mock()
    mock_message.content = [mock_text_block, mock_tool_use_block]
    mock_message.usage = Mock(input_tokens=15, output_tokens=25)
    
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_message
    mock_anthropic_module.Anthropic.return_value = mock_client
    
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        router = LlmRouter()
        agent = _anthropic_agent(tools=["web_search"])
        
        result = router.anthropic_complete(
            agent,
            system="test system",
            messages=[{"role": "user", "content": "search for something"}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}]
        )
        
        # Result should contain both text and tool_use blocks
        assert result == mock_message
        assert len(result.content) == 2
        assert result.content[0].text == "Here's some information"
        assert result.content[1].type == "tool_use"
