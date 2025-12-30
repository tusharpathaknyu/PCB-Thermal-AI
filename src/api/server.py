"""
FastAPI Server for PCB Thermal Prediction

Provides REST API for:
- Uploading PCB features and getting temperature predictions
- Health checks
- Model info

Run with: uvicorn src.api.server:app --reload
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import io
import base64
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import ThermalPredictor

# Initialize app
app = FastAPI(
    title="PCB Thermal AI API",
    description="ML-based PCB temperature prediction from layout features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (loaded on startup)
predictor: Optional[ThermalPredictor] = None


# ===== Pydantic Models =====

class PCBFeatures(BaseModel):
    """Input PCB features for prediction"""
    copper: List[List[float]] = Field(..., description="Copper density map (H x W), values 0-1")
    vias: List[List[float]] = Field(..., description="Via location map (H x W), values 0-1")
    components: List[List[float]] = Field(..., description="Component footprint map (H x W), values 0-1")
    power: List[List[float]] = Field(..., description="Power dissipation map (H x W), normalized")

class PredictionResponse(BaseModel):
    """Temperature prediction response"""
    temperature: List[List[float]] = Field(..., description="Temperature field (H x W) in °C")
    max_temp: float = Field(..., description="Maximum temperature in °C")
    min_temp: float = Field(..., description="Minimum temperature in °C")
    mean_temp: float = Field(..., description="Mean temperature in °C")
    hotspot_location: List[int] = Field(..., description="[y, x] location of hotspot")
    temp_range: float = Field(..., description="Temperature range in °C")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    parameters: int
    input_shape: List[int]
    output_shape: List[int]
    normalization_stats: Dict[str, float]


# ===== Startup/Shutdown Events =====

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    
    checkpoint_path = Path("checkpoints/best.pth")
    
    if checkpoint_path.exists():
        try:
            predictor = ThermalPredictor.load(str(checkpoint_path))
            print(f"✓ Model loaded from {checkpoint_path}")
            print(f"  Device: {predictor.device}")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            predictor = None
    else:
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        print("  Run training first: python scripts/train.py")
        predictor = None


# ===== API Endpoints =====

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "PCB Thermal AI API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "none"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about loaded model"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=predictor.model.__class__.__name__,
        parameters=sum(p.numel() for p in predictor.model.parameters()),
        input_shape=[4, 128, 128],
        output_shape=[1, 128, 128],
        normalization_stats=predictor.output_stats
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: PCBFeatures):
    """
    Predict temperature field from PCB features.
    
    Input arrays should be 128x128 (or will be resized).
    All values should be normalized to 0-1 range.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy arrays
        copper = np.array(features.copper, dtype=np.float32)
        vias = np.array(features.vias, dtype=np.float32)
        components = np.array(features.components, dtype=np.float32)
        power = np.array(features.power, dtype=np.float32)
        
        # Validate shapes
        expected_shape = (128, 128)
        for name, arr in [("copper", copper), ("vias", vias), 
                          ("components", components), ("power", power)]:
            if arr.shape != expected_shape:
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} has shape {arr.shape}, expected {expected_shape}"
                )
        
        # Run prediction
        result = predictor.predict(
            copper=copper,
            vias=vias,
            components=components,
            power=power,
            return_dict=True
        )
        
        return PredictionResponse(
            temperature=result['temperature'].tolist(),
            max_temp=result['max_temp'],
            min_temp=result['min_temp'],
            mean_temp=result['mean_temp'],
            hotspot_location=list(result['hotspot_location']),
            temp_range=result['temp_range']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/quick", tags=["Prediction"])
async def predict_quick(
    total_power: float = 2.0,
    copper_fill: float = 0.5,
    num_components: int = 5
):
    """
    Quick prediction with synthetic PCB generation.
    
    Generates a random PCB layout based on parameters and predicts temperature.
    Useful for testing without providing full feature maps.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Import PCB generator
        from src.data_generation import PCBGenerator
        
        # Generate random PCB
        generator = PCBGenerator(grid_size=(128, 128))
        layout = generator.generate(
            complexity="medium",
            num_components=num_components,
            total_power=total_power
        )
        
        # Adjust copper fill
        layout.copper_density = layout.copper_density * (copper_fill / layout.copper_density.mean())
        layout.copper_density = np.clip(layout.copper_density, 0, 1)
        
        # Run prediction
        result = predictor.predict(
            copper=layout.copper_density,
            vias=layout.via_map,
            components=layout.component_map,
            power=layout.power_map / 1000,  # Normalize
            return_dict=True
        )
        
        return {
            "prediction": {
                "max_temp": result['max_temp'],
                "min_temp": result['min_temp'],
                "mean_temp": result['mean_temp'],
                "hotspot_location": list(result['hotspot_location']),
            },
            "input_summary": {
                "total_power": total_power,
                "copper_fill": float(layout.copper_density.mean()),
                "num_components": len(layout.components),
                "num_vias": len(layout.vias)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image", tags=["Prediction"])
async def predict_from_npz(file: UploadFile = File(...), sample_idx: int = 0):
    """
    Predict from uploaded .npz file.
    
    Upload a .npz file containing 'inputs' array with shape (N, 4, H, W).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.npz'):
        raise HTTPException(status_code=400, detail="File must be .npz format")
    
    try:
        # Read file
        contents = await file.read()
        data = np.load(io.BytesIO(contents))
        
        if 'inputs' not in data:
            raise HTTPException(status_code=400, detail="File must contain 'inputs' array")
        
        inputs = data['inputs']
        
        if sample_idx >= len(inputs):
            raise HTTPException(
                status_code=400, 
                detail=f"sample_idx {sample_idx} out of range (file has {len(inputs)} samples)"
            )
        
        sample = inputs[sample_idx]
        
        # Predict
        result = predictor.predict(
            copper=sample[0],
            vias=sample[1],
            components=sample[2],
            power=sample[3],
            return_dict=True
        )
        
        response = {
            "max_temp": result['max_temp'],
            "min_temp": result['min_temp'],
            "mean_temp": result['mean_temp'],
            "hotspot_location": list(result['hotspot_location']),
            "temp_range": result['temp_range']
        }
        
        # Include ground truth if available
        if 'outputs' in data:
            gt = data['outputs'][sample_idx, 0]
            response['ground_truth_max'] = float(gt.max())
            response['mae'] = float(np.abs(result['temperature'] - gt).mean())
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Run Server =====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
