"""
Architect Agent MCP Server

This module implements the MCP server for the Architect agent.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from factory.core.mcp_server import MCPServer, MCPMessage
from factory.core.competence import AgentProfile

logger = logging.getLogger(__name__)


class ArchitectMCPServer(MCPServer):
    """MCP server for the Architect agent."""
    
    def __init__(self, agent_profile: AgentProfile):
        super().__init__(
            agent_name=agent_profile.name,
            agent_type="architect"
        )
        self.agent_profile = agent_profile
        self.current_plan: Dict[str, Any] = {}
        
    def _process(self) -> None:
        """Architect-specific processing logic."""
        # Check if there is an active plan and process it
        if self.current_plan:
            plan_id = self.current_plan.get("plan_id")
            status = self.current_plan.get("status")
            
            if status == "planning":
                # Simulate planning process
                logger.info(f"Processing plan {plan_id}")
                # Update status to indicate planning is complete
                self.current_plan["status"] = "planning_complete"
                
                # Send a notification or update to indicate completion
                completion_message = MCPMessage(
                    message_type="status",
                    payload={"plan_id": plan_id, "status": "planning_complete"},
                    sender=self.agent_name
                )
                self.send_message(completion_message)
    
    def _handle_task_message(self, payload: Dict[str, Any]) -> None:
        """Handle a task message for the Architect agent."""
        plan_id = payload.get("plan_id")
        issue_iid = payload.get("issue_iid")
        
        logger.info(f"Architect agent received planning task {plan_id} for issue {issue_iid}")
        self.current_plan = {
            "plan_id": plan_id,
            "issue_iid": issue_iid,
            "status": "planning"
        }
        
        # Here you would typically:
        # 1. Parse the planning details
        # 2. Execute the architect agent logic
        # 3. Update plan status
        # 4. Send completion notification
        
    def _handle_status_message(self, payload: Dict[str, Any]) -> None:
        """Handle a status message for the Architect agent."""
        status = payload.get("status")
        logger.info(f"Architect agent status update: {status}")
        
        # Update internal state based on status
        if self.current_plan:
            self.current_plan["status"] = status
            
    def _handle_control_message(self, payload: Dict[str, Any]) -> None:
        """Handle a control message for the Architect agent."""
        command = payload.get("command")
        logger.info(f"Architect agent control command: {command}")
        
        if command == "stop":
            self.stop()
        elif command == "pause":
            # Implement pause logic
            if self.current_plan:
                self.current_plan["status"] = "paused"
                logger.info(f"Plan {self.current_plan.get('plan_id')} paused")
        elif command == "resume":
            # Implement resume logic
            if self.current_plan:
                self.current_plan["status"] = "planning"
                logger.info(f"Plan {self.current_plan.get('plan_id')} resumed")
