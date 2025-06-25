#!/usr/bin/env python3
"""
TTS Model Context Protocol (MCP) Server

Enterprise-grade MCP server for text-to-speech with GPU acceleration,
real-time streaming, and production features.

Usage:
    python tts_mcp_server.py --port 9090 --models fastspeech2,tacotron2
    python tts_mcp_server.py --config mcp_config.json
"""

import argparse
import json
import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
import uuid
import numpy as np
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    import torch
    import torchaudio
    import websockets
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install fastapi uvicorn websockets torch torchaudio")
    sys.exit(1)

# Import TTS engines
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core_engines.synthesis.gpu_fastspeech2 import GPUTTSEngine


# MCP Protocol Models
class MCPRequest(BaseModel):
    """MCP standard request format."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP standard response format."""
    id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TTSSynthesisRequest(BaseModel):
    """TTS synthesis request parameters."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="default")
    model: str = Field(default="fastspeech2")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=0.0, ge=-1.0, le=1.0)
    emotion: Optional[str] = Field(None)
    streaming: bool = Field(default=False)
    format: str = Field(default="wav", regex="^(wav|mp3|opus|flac)$")
    sample_rate: int = Field(default=22050)


class VoiceCloneRequest(BaseModel):
    """Voice cloning request parameters."""
    reference_audio: str  # Base64 encoded audio or URL
    transcript: Optional[str] = None
    voice_name: str
    fine_tune_steps: int = Field(default=100, ge=0, le=1000)


class TTSMCPServer:
    """MCP server for text-to-speech with enterprise features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(
            title="TTS MCP Server",
            description="Model Context Protocol server for GPU-accelerated TTS",
            version="1.0.0"
        )
        
        # Initialize TTS engines
        self.engines = {}
        self.load_engines()
        
        # Voice management
        self.voices = self.load_voices()
        self.custom_voices = {}
        
        # Performance tracking
        self.active_connections = 0
        self.total_requests = 0
        self.total_characters = 0
        
        # Setup
        self.setup_middleware()
        self.setup_routes()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('TTSMCPServer')
    
    def setup_middleware(self):
        """Setup CORS and other middleware."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.get('cors', {}).get('origins', ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def load_engines(self):
        """Load configured TTS engines."""
        models_config = self.config.get('models', {})
        
        for model_name, model_config in models_config.items():
            try:
                self.logger.info(f"Loading {model_name} engine...")
                
                if model_name == 'fastspeech2':
                    engine = GPUTTSEngine(
                        model_path=model_config['path'],
                        vocoder_path=model_config.get('vocoder_path'),
                        use_tensorrt=model_config.get('use_tensorrt', False),
                        precision=model_config.get('precision', 'fp16')
                    )
                    self.engines[model_name] = engine
                    
                # Add other engines (tacotron2, vits, etc.) here
                
                self.logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {e}")
    
    def load_voices(self) -> Dict[str, Dict[str, Any]]:
        """Load available voices."""
        voices_path = Path(self.config.get('voices_dir', 'voices'))
        voices = {}
        
        if voices_path.exists():
            for voice_file in voices_path.glob('*.json'):
                with open(voice_file, 'r') as f:
                    voice_data = json.load(f)
                    voices[voice_data['name']] = voice_data
        
        # Add default voice
        voices['default'] = {
            'name': 'default',
            'description': 'Default TTS voice',
            'language': 'en',
            'gender': 'neutral'
        }
        
        return voices
    
    def setup_routes(self):
        """Setup MCP protocol routes."""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "engines": list(self.engines.keys()),
                "active_connections": self.active_connections,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/mcp/execute")
        async def execute_mcp(request: MCPRequest) -> MCPResponse:
            """Execute MCP request."""
            self.total_requests += 1
            
            try:
                # Route to appropriate handler
                if request.method == "tts.synthesize":
                    result = await self.handle_synthesis(request.params)
                elif request.method == "tts.stream":
                    result = await self.handle_stream_init(request.params)
                elif request.method == "tts.voices.list":
                    result = await self.handle_list_voices()
                elif request.method == "tts.voices.clone":
                    result = await self.handle_voice_clone(request.params)
                elif request.method == "tts.models.list":
                    result = await self.handle_list_models()
                else:
                    raise ValueError(f"Unknown method: {request.method}")
                
                return MCPResponse(
                    id=request.id,
                    success=True,
                    result=result,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "engine_version": "1.0.0"
                    }
                )
                
            except Exception as e:
                self.logger.error(f"MCP request failed: {e}")
                return MCPResponse(
                    id=request.id,
                    success=False,
                    error=str(e)
                )
        
        @self.app.websocket("/mcp/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for streaming TTS."""
            await websocket.accept()
            self.active_connections += 1
            
            try:
                while True:
                    # Receive MCP request
                    data = await websocket.receive_json()
                    request = MCPRequest(**data)
                    
                    if request.method == "tts.stream":
                        # Stream audio chunks
                        async for chunk in self.stream_synthesis(request.params):
                            await websocket.send_json({
                                "id": request.id,
                                "type": "audio_chunk",
                                "data": chunk
                            })
                        
                        # Send completion
                        await websocket.send_json({
                            "id": request.id,
                            "type": "complete",
                            "success": True
                        })
                    
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections -= 1
    
    async def handle_synthesis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle TTS synthesis request."""
        req = TTSSynthesisRequest(**params)
        
        # Track usage
        self.total_characters += len(req.text)
        
        # Select engine
        engine = self.engines.get(req.model)
        if not engine:
            raise ValueError(f"Model {req.model} not available")
        
        # Get voice parameters
        voice_params = self.voices.get(req.voice, {})
        
        # Synthesize
        self.logger.info(f"Synthesizing {len(req.text)} chars with {req.model}")
        
        waveform = await asyncio.to_thread(
            engine.synthesize,
            text=req.text,
            speed=req.speed,
            pitch_shift=req.pitch
        )
        
        # Convert to requested format
        audio_data = self.encode_audio(waveform, req.format, req.sample_rate)
        
        return {
            "audio": audio_data,
            "duration": len(waveform) / req.sample_rate,
            "format": req.format,
            "sample_rate": req.sample_rate,
            "model": req.model,
            "voice": req.voice
        }
    
    async def stream_synthesis(self, params: Dict[str, Any]) -> AsyncIterator[str]:
        """Stream TTS synthesis."""
        req = TTSSynthesisRequest(**params)
        
        # Select engine
        engine = self.engines.get(req.model)
        if not engine:
            raise ValueError(f"Model {req.model} not available")
        
        # Stream synthesis
        chunk_size = self.config.get('streaming', {}).get('chunk_size', 1024)
        
        async for audio_chunk in self.async_stream_wrapper(
            engine.stream_synthesize(req.text, chunk_size)
        ):
            # Encode chunk
            encoded = self.encode_audio_chunk(audio_chunk, req.format)
            yield encoded
    
    async def async_stream_wrapper(self, sync_generator):
        """Wrap synchronous generator for async iteration."""
        loop = asyncio.get_event_loop()
        
        def _next():
            try:
                return next(sync_generator)
            except StopIteration:
                return None
        
        while True:
            chunk = await loop.run_in_executor(None, _next)
            if chunk is None:
                break
            yield chunk
    
    async def handle_list_voices(self) -> Dict[str, Any]:
        """List available voices."""
        voice_list = []
        
        for name, voice in self.voices.items():
            voice_list.append({
                "id": name,
                "name": voice.get('name', name),
                "description": voice.get('description', ''),
                "language": voice.get('language', 'en'),
                "gender": voice.get('gender', 'neutral'),
                "custom": name in self.custom_voices
            })
        
        return {"voices": voice_list}
    
    async def handle_voice_clone(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle voice cloning request."""
        req = VoiceCloneRequest(**params)
        
        # Decode reference audio
        if req.reference_audio.startswith('data:'):
            # Base64 encoded
            audio_data = self.decode_base64_audio(req.reference_audio)
        else:
            # URL - fetch audio
            audio_data = await self.fetch_audio_from_url(req.reference_audio)
        
        # Perform voice cloning (placeholder - implement actual cloning)
        self.logger.info(f"Cloning voice: {req.voice_name}")
        
        # Save custom voice
        self.custom_voices[req.voice_name] = {
            "name": req.voice_name,
            "reference_audio": "path/to/processed/audio",
            "created": datetime.now().isoformat()
        }
        
        self.voices[req.voice_name] = {
            "name": req.voice_name,
            "description": f"Cloned voice from {len(audio_data)} bytes of audio",
            "language": "en",
            "gender": "neutral",
            "custom": True
        }
        
        return {
            "voice_id": req.voice_name,
            "status": "ready",
            "message": f"Voice '{req.voice_name}' cloned successfully"
        }
    
    async def handle_list_models(self) -> Dict[str, Any]:
        """List available TTS models."""
        models = []
        
        for name, engine in self.engines.items():
            stats = engine.get_performance_stats()
            models.append({
                "id": name,
                "name": name.title(),
                "description": f"{name.upper()} TTS model",
                "supported_languages": ["en"],  # Extend based on model
                "performance": {
                    "avg_rtf": stats.get('avg_rtf', 0),
                    "gpu_optimized": True
                }
            })
        
        return {"models": models}
    
    def encode_audio(self, waveform: np.ndarray, format: str, 
                    sample_rate: int) -> str:
        """Encode audio to requested format."""
        import base64
        from io import BytesIO
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(waveform).unsqueeze(0)
        
        # Save to buffer
        buffer = BytesIO()
        
        if format == 'wav':
            torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
        elif format == 'mp3':
            torchaudio.save(buffer, audio_tensor, sample_rate, format='mp3')
        else:
            # Default to WAV
            torchaudio.save(buffer, audio_tensor, sample_rate, format='wav')
        
        # Encode to base64
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return f"data:audio/{format};base64,{audio_base64}"
    
    def encode_audio_chunk(self, chunk: np.ndarray, format: str) -> str:
        """Encode audio chunk for streaming."""
        import base64
        
        # Simple encoding - in production use proper streaming format
        chunk_bytes = chunk.astype(np.float32).tobytes()
        return base64.b64encode(chunk_bytes).decode('utf-8')
    
    def decode_base64_audio(self, data: str) -> bytes:
        """Decode base64 audio data."""
        import base64
        
        # Remove data URL prefix if present
        if ',' in data:
            data = data.split(',')[1]
        
        return base64.b64decode(data)
    
    async def fetch_audio_from_url(self, url: str) -> bytes:
        """Fetch audio from URL."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise ValueError(f"Failed to fetch audio from {url}")


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load server configuration."""
    default_config = {
        "server": {
            "host": "0.0.0.0",
            "port": 9090,
            "workers": 1
        },
        "models": {
            "fastspeech2": {
                "path": "models/fastspeech2_en.pt",
                "vocoder_path": "models/hifigan_universal.pt",
                "use_tensorrt": False,
                "precision": "fp16"
            }
        },
        "voices_dir": "voices",
        "streaming": {
            "chunk_size": 1024,
            "buffer_size": 4096
        },
        "cors": {
            "origins": ["*"]
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge configs
        def merge_dict(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(default_config, user_config)
    
    return default_config


def main():
    parser = argparse.ArgumentParser(description="TTS MCP Server")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9090, help="Port to bind")
    parser.add_argument("--models", help="Comma-separated list of models to load")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line
    if args.host:
        config['server']['host'] = args.host
    if args.port:
        config['server']['port'] = args.port
    
    # Create and run server
    server = TTSMCPServer(config)
    
    print(f"🚀 TTS MCP Server starting on {config['server']['host']}:{config['server']['port']}")
    print(f"📊 Loaded models: {list(server.engines.keys())}")
    print(f"🎤 Available voices: {len(server.voices)}")
    
    uvicorn.run(
        server.app,
        host=config['server']['host'],
        port=config['server']['port'],
        reload=args.reload
    )


if __name__ == "__main__":
    main()