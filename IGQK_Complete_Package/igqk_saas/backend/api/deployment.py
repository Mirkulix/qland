"""
Deployment API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

router = APIRouter()


class DeploymentTarget(str, Enum):
    """Deployment target options"""
    IGQK_API = "igqk_api"
    AWS_SAGEMAKER = "aws_sagemaker"
    GOOGLE_VERTEX = "google_vertex"
    AZURE_ML = "azure_ml"
    CUSTOM_API = "custom_api"


class DeploymentConfig(BaseModel):
    """Deployment configuration"""
    model_id: str
    name: str
    target: DeploymentTarget
    region: Optional[str] = "us-east-1"
    instance_type: Optional[str] = "t4"
    auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10


class Deployment(BaseModel):
    """Deployed model"""
    id: str
    model_id: str
    name: str
    status: str  # deploying, running, stopped, failed
    endpoint_url: Optional[str]
    requests_count: int
    avg_latency_ms: float
    created_at: str


import uuid
import time
from typing import Dict

deployments_store: Dict[str, Deployment] = {}


@router.post("/deploy")
async def deploy_model(config: DeploymentConfig):
    """Deploy a model to target environment"""
    # Only this server can serve (igqk_api); the cloud targets are not connected.
    if config.target != DeploymentTarget.IGQK_API:
        raise HTTPException(status_code=501, detail=f"deployment target '{config.target.value}' is not implemented; use 'igqk_api' (this server)")
    # Only a model that can answer `predict` is deployed; the endpoint is this server's.
    from api.models import servable
    ok, info = servable(config.model_id)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in info else 422, detail=f"cannot deploy: {info}")
    dep_id = f"dep_{uuid.uuid4().hex[:8]}"
    endpoint = f"/api/models/{config.model_id}/predict"

    dep = Deployment(
        id=dep_id,
        model_id=config.model_id,
        name=config.name,
        status="running",
        endpoint_url=endpoint,
        requests_count=0,
        avg_latency_ms=0.0,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    deployments_store[dep_id] = dep

    return {
        "deployment_id": dep_id,
        "status": "running",
        "message": f"Model '{config.name}' is served by this server at {endpoint}",
        "endpoint_url": endpoint,
        "monitor_url": f"/api/deployment/{dep_id}"
    }


@router.get("/{deployment_id}", response_model=Deployment)
async def get_deployment_status(deployment_id: str):
    """Get deployment status"""
    if deployment_id not in deployments_store:
        raise HTTPException(status_code=404, detail=f"Deployment '{deployment_id}' not found")
    return deployments_store[deployment_id]


@router.get("/", response_model=List[Deployment])
async def list_deployments():
    """List all active deployments"""
    return list(deployments_store.values())


@router.delete("/{deployment_id}")
async def stop_deployment(deployment_id: str):
    """Stop a deployment"""
    if deployment_id not in deployments_store:
        raise HTTPException(status_code=404, detail=f"Deployment '{deployment_id}' not found")
    deployments_store[deployment_id].status = "stopped"
    return {
        "deployment_id": deployment_id,
        "status": "stopped",
        "message": "Deployment stopped successfully"
    }
