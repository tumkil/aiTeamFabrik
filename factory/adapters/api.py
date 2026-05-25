"""
REST API adapter for SoftwareTeamFabrik using FastAPI.

This module provides a REST API layer to enable GUI clients to interact with
the SoftwareTeamFabrik system, exposing factory commands and log streaming
capabilities.
"""

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import json

# Local imports
from factory.commands.chat import cmd_po

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SoftwareTeamFabrik API",
    description="REST API for interacting with the SoftwareTeamFabrik system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Create API router
api_router = APIRouter(prefix="/api/v1")


# Models
class CommandResponse(BaseModel):
    """Response model for command execution results"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CommandRequest(BaseModel):
    """Request model for command execution"""
    command: str
    args: Optional[Dict[str, Any]] = None


# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "SoftwareTeamFabrik API is running"
    }


# Status endpoint
@api_router.get("/status")
async def get_status():
    """Get system status"""
    return CommandResponse(
        success=True,
        message="Status retrieved successfully",
        data={"status": "operational"}
    ).dict()


# Command execution endpoint
@api_router.post("/run")
async def run_command(request: CommandRequest):
    """Execute a factory command"""
    try:
        if request.command == "po":
            # Execute PO command
            result = cmd_po()
            return CommandResponse(
                success=True,
                message="Command executed successfully",
                data={"result": result}
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown command: {request.command}"
            )
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Command execution failed: {e}"
        )


# Logs streaming endpoint
@api_router.get("/logs")
async def stream_logs():
    """Stream system logs"""
    async def log_generator():
        # Simulate log streaming
        # In a real implementation, this would stream actual logs
        yield "data: " + json.dumps({"log": "System log entry 1"}) + "\n\n"
        yield "data: " + json.dumps({"log": "System log entry 2"}) + "\n\n"
        yield "data: " + json.dumps({"log": "System log entry 3"}) + "\n\n"
    
    return StreamingResponse(log_generator(), media_type="text/event-stream")


# Include API router
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
