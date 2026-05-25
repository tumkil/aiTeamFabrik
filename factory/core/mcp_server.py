"""
MCP (Modular Control Protocol) Server Base Class

This module provides the base MCP server functionality that all agent-specific
MCP servers will inherit from. Each agent type will have its own MCP server
implementation for better modularity and extensibility.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """A message in the MCP protocol."""
    message_type: str
    payload: Dict[str, Any]
    sender: Optional[str] = None
    timestamp: Optional[str] = None


class MCPServer(ABC):
    """Base class for all MCP servers."""
    
    def __init__(self, agent_name: str, agent_type: str):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._message_queue: list[MCPMessage] = []
        self._lock = threading.Lock()
        
    def start(self) -> None:
        """Start the MCP server in a background thread."""
        if self._running:
            logger.warning(f"MCP server for {self.agent_name} is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            name=f"MCP-{self.agent_type}-{self.agent_name}",
            daemon=True
        )
        self._thread.start()
        logger.info(f"Started MCP server for {self.agent_type} agent: {self.agent_name}")
        
    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning(f"MCP server for {self.agent_name} did not stop gracefully")
        logger.info(f"Stopped MCP server for {self.agent_type} agent: {self.agent_name}")
        
    def send_message(self, message: MCPMessage) -> None:
        """Send a message to this MCP server."""
        with self._lock:
            self._message_queue.append(message)
        
    def _process_message(self, message: MCPMessage) -> None:
        """Process an incoming MCP message."""
        try:
            if message.message_type == "task":
                self._handle_task_message(message.payload)
            elif message.message_type == "status":
                self._handle_status_message(message.payload)
            elif message.message_type == "control":
                self._handle_control_message(message.payload)
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
        except Exception as e:
            logger.error(f"Error processing message {message.message_type}: {e}")
            
    def _run_server(self) -> None:
        """Main server loop."""
        logger.info(f"MCP server loop started for {self.agent_name}")
        while self._running:
            try:
                # Process messages
                with self._lock:
                    if self._message_queue:
                        message = self._message_queue.pop(0)
                        self._process_message(message)
                
                # Agent-specific processing
                self._process()
                
                # Small sleep to prevent busy waiting
                threading.Event().wait(0.1)
                
            except Exception as e:
                logger.error(f"Error in MCP server loop for {self.agent_name}: {e}")
                threading.Event().wait(1.0)  # Backoff on error
        
        logger.info(f"MCP server loop stopped for {self.agent_name}")
        
    @abstractmethod
    def _process(self) -> None:
        """Agent-specific processing logic. Called in the main server loop."""
        pass
        
    @abstractmethod
    def _handle_task_message(self, payload: Dict[str, Any]) -> None:
        """Handle a task message."""
        pass
        
    @abstractmethod
    def _handle_status_message(self, payload: Dict[str, Any]) -> None:
        """Handle a status message."""
        pass
        
    @abstractmethod
    def _handle_control_message(self, payload: Dict[str, Any]) -> None:
        """Handle a control message."""
        pass
