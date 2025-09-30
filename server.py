import time
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import torchaudio
from io import BytesIO
import os
from dotenv import load_dotenv
import ffmpeg
import asyncio
import sys
from pydub import AudioSegment
import io
import numpy as np


load_dotenv()

PORT = int(os.getenv('PORT', 5000))

# Add the source path for chatterbox imports
CHATTERBOX_SRC_PATH = os.getenv('CHATTERBOX_SRC_PATH', '/home/incode/projects/chatterbox2/chatterbox-streaming/src')
sys.path.insert(0, CHATTERBOX_SRC_PATH)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Initialize the FastAPI app
app = FastAPI()
app.config = {"STORAGE_FOLDER": "./storage"}

cwd = os.getcwd()

def load_lora_model(lora_checkpoint_path: str, device: str = "cuda"):
    """
    Load ChatterboxMultilingualTTS base model and merge LoRA weights.
    Args:
        lora_checkpoint_path (str): Path to the LoRA checkpoint (.pt file)
        device (str): 'cuda' or 'cpu'
    Returns:
        lora_model (ChatterboxMultilingualTTS): Model with LoRA applied
    """
    # Load base multilingual model
    print("Loading base multilingual model...")
    lora_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    
    # Load LoRA checkpoint
    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} ...")
    checkpoint = torch.load(lora_checkpoint_path, map_location=device)
    if "lora_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'lora_state_dict' key")
    lora_state_dict = checkpoint["lora_state_dict"]
    
    # Group LoRA A and B matrices
    lora_layers = {}
    for key, weight in lora_state_dict.items():
        if ".lora_A" in key:
            base_name = key.replace(".lora_A", "")
            lora_layers.setdefault(base_name, {})["A"] = weight
        elif ".lora_B" in key:
            base_name = key.replace(".lora_B", "")
            lora_layers.setdefault(base_name, {})["B"] = weight
    
    # Merge LoRA into base model
    modified_state = lora_model.t3.state_dict().copy()
    applied_count = 0
    for base_name, weights in lora_layers.items():
        if "A" in weights and "B" in weights:
            param_name = f"tfmr.{base_name}.weight"
            if param_name in modified_state:
                original_weight = modified_state[param_name]
                lora_A = weights["A"].to(original_weight.device, original_weight.dtype)
                lora_B = weights["B"].to(original_weight.device, original_weight.dtype)
                delta_W = torch.mm(lora_B, lora_A)
                if delta_W.shape == original_weight.shape:
                    modified_state[param_name] = original_weight + delta_W
                    applied_count += 1
    
    # Load merged weights into model
    lora_model.t3.load_state_dict(modified_state)
    print(f"✅ Applied LoRA to {applied_count} layers")
    return lora_model

# Load the ChatterBox model
print("Loading ChatterBox model...")
LORA_CHECKPOINT_PATH = "/home/incode/projects/chatterbox-streaming/checkpoint_epoch38_step24000.pt"

if LORA_CHECKPOINT_PATH and os.path.exists(LORA_CHECKPOINT_PATH):
    print(f"Loading model with LoRA from {LORA_CHECKPOINT_PATH}")
    model = load_lora_model(LORA_CHECKPOINT_PATH, device="cuda")
else:
    print("Loading base multilingual model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(2)  # Adjust based on GPU capacity

# Route to synthesize speech with voice cloning support
@app.post('/stream/audio/speech')
async def synthesize(request: Request):
    data = await request.json()
    input_text = data.get('input', '')
    language_id = data.get('language_id', 'ar')
    audio_prompt_path = data.get('voice_id')
    speed = data.get('speed', 1.0)
    exaggeration = data.get('exaggeration', 0.6)
    cfg_weight = data.get('cfg_weight', 0.5)
    temperature = data.get('temperature', 0.8)
    output_sample_rate = data.get('output_sample_rate', 8000)
    chunk_size = data.get('chunk_size', 50)

    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if audio_prompt_path:
        audio_prompt_path = os.path.join(cwd, audio_prompt_path + '.wav')

    # Validate audio prompt path if provided
    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        raise HTTPException(status_code=400, detail=f"Audio prompt file not found: {audio_prompt_path}")

    # Check if we need to store the audio in a file
    store_file = request.query_params.get('type') == 'file'
    file_name = request.query_params.get('name', 'output.wav') if store_file else None

    print('Received stream data:', data)

    async def generate_chunks():
        async with semaphore:
            t0 = time.time()
            
            all_chunks = []
            chunk_count = 0

            # Prepare generation parameters
            generation_params = {
                'text': input_text,
                'language_id': language_id,
                'exaggeration': exaggeration,
                'cfg_weight': cfg_weight,
                'temperature': temperature,
                'chunk_size': chunk_size,
                'print_metrics': True,
                'context_window': 10,
                'fade_duration': 0.05,
                'max_new_tokens': int(len(input_text)*2 + 10)
            }

            # Add audio prompt if provided (for voice cloning)
            if audio_prompt_path:
                generation_params['audio_prompt_path'] = audio_prompt_path

            # Use ChatterBox streaming generation
            for audio_chunk, metrics in model.generate_stream(**generation_params):
                if chunk_count == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                    if metrics.latency_to_first_chunk:
                        print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
                
                # Move tensor to CPU and convert to numpy array
                chunk_cpu = audio_chunk.cpu().detach().numpy()

                if store_file:
                    all_chunks.append(audio_chunk)

                # Convert numpy array to AudioSegment
                # Ensure the audio is in the correct format (float32 to int16)
                chunk_int16 = (chunk_cpu * 32767).astype(np.int16)
                
                # Create AudioSegment from raw data
                audio_segment = AudioSegment(
                    data=chunk_int16.tobytes(),
                    sample_width=2,  # 16-bit = 2 bytes
                    frame_rate=model.sr,
                    channels=1
                )
                
                # Resample to output sample rate if different
                if output_sample_rate != model.sr:
                    audio_segment = audio_segment.set_frame_rate(output_sample_rate)
                
                # Export to mu-law format
                buffer = io.BytesIO()
                audio_segment.export(buffer, format='mulaw', codec='pcm_mulaw')
                yield buffer.getvalue()
                chunk_count += 1

            # Save to file if required
            if store_file and all_chunks:
                print(f"Number of chunks to combine: {len(all_chunks)}")
                
                # Debug: Check individual chunk shapes before concatenation
                for i, chunk in enumerate(all_chunks):
                    print(f"Chunk {i} shape before concatenation: {chunk.shape}")
                
                combined_audio = torch.cat(all_chunks, dim=-1)
                print(f"Combined audio shape after concatenation: {combined_audio.shape}")
                print(f"Combined audio dimensions: {combined_audio.ndim}D")
                
                # Check if we need to adjust dimensions
                if combined_audio.ndim == 1:
                    print("Converting 1D tensor to 2D by adding channel dimension")
                    combined_audio = combined_audio.unsqueeze(0)
                    print(f"Combined audio shape after unsqueeze: {combined_audio.shape}")
                elif combined_audio.ndim == 3:
                    print("Warning: 3D tensor detected, squeezing first dimension")
                    combined_audio = combined_audio.squeeze(0)
                    print(f"Combined audio shape after squeeze: {combined_audio.shape}")
                
                print(f"Final tensor shape for torchaudio.save: {combined_audio.shape}")
                print(f"Expected format: (channels, samples) - should be 2D")
                torchaudio.save(
                    os.path.join(app.config['STORAGE_FOLDER'], file_name), 
                    combined_audio, 
                    model.sr
                )
                print(f"Audio saved to {os.path.join(app.config['STORAGE_FOLDER'], file_name)}")

    return StreamingResponse(generate_chunks(), media_type='audio/x-mulaw')

# Route for non-streaming inference with voice cloning support
@app.post('/audio/speech')
async def generate_audio(request: Request):
    data = await request.json()
    input_text = data.get('input', '')
    language_id = data.get('language_id', 'ar')
    audio_prompt_path = data.get('audio_prompt_path')
    speed = data.get('speed', 1.0)
    exaggeration = data.get('exaggeration', 0.6)
    cfg_weight = data.get('cfg_weight', 0.5)
    temperature = data.get('temperature', 0.8)
    chunk_size = data.get('chunk_size', 50)

    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Validate audio prompt path if provided
    if audio_prompt_path and not os.path.exists(audio_prompt_path):
        raise HTTPException(status_code=400, detail=f"Audio prompt file not found: {audio_prompt_path}")

    async def generate_audio_stream():
        # Collect all chunks for non-streaming response
        audio_chunks = []
        
        # Prepare generation parameters
        generation_params = {
            'text': input_text,
            'language_id': language_id,
            'exaggeration': exaggeration,
            'cfg_weight': cfg_weight,
            'temperature': temperature,
            'chunk_size': chunk_size,
            'print_metrics': True,
            'context_window': 100,
            'fade_duration': 1,
            'max_new_tokens': int(len(input_text)*2 + 10)
        }

        # Add audio prompt if provided (for voice cloning)
        if audio_prompt_path:
            generation_params['audio_prompt_path'] = audio_prompt_path

        for audio_chunk, metrics in model.generate_stream(**generation_params):
            audio_chunks.append(audio_chunk)

        # Combine all chunks into final audio
        if audio_chunks:
            final_audio = torch.cat(audio_chunks, dim=-1)
            
            # Convert to WAV bytes
            wav_buffer = BytesIO()
            torchaudio.save(wav_buffer, final_audio.unsqueeze(0), model.sr, format='wav')
            wav_buffer.seek(0)

            # Stream WAV audio
            while chunk := wav_buffer.read(1024):
                yield chunk

    return StreamingResponse(generate_audio_stream(), media_type='audio/wav')

# Route for voice cloning with file upload
@app.post('/voice-clone/stream')
async def voice_clone_stream(request: Request):
    """Enhanced voice cloning endpoint with streaming"""
    data = await request.json()
    input_text = data.get('input', '')
    language_id = data.get('language_id', 'ar')
    audio_prompt_path = data.get('audio_prompt_path')
    exaggeration = data.get('exaggeration', 0.6)
    cfg_weight = data.get('cfg_weight', 0.3)
    temperature = data.get('temperature', 0.8)
    chunk_size = data.get('chunk_size', 25)
    output_sample_rate = data.get('output_sample_rate', 8000)

    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if not audio_prompt_path:
        raise HTTPException(status_code=400, detail="Audio prompt path required for voice cloning")
    
    if not os.path.exists(audio_prompt_path):
        raise HTTPException(status_code=400, detail=f"Audio prompt file not found: {audio_prompt_path}")

    # Check if we need to store the audio in a file
    store_file = request.query_params.get('type') == 'file'
    file_name = request.query_params.get('name', 'voice_clone_output.wav') if store_file else None

    print('Voice cloning request:', data)

    async def generate_voice_clone_chunks():
        async with semaphore:
            t0 = time.time()
            all_chunks = []
            chunk_count = 0

            for audio_chunk, metrics in model.generate_stream(
                text=input_text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                chunk_size=chunk_size,
                language_id=language_id,
                print_metrics=True
            ):
                if chunk_count == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                    if hasattr(metrics, 'latency_to_first_chunk') and metrics.latency_to_first_chunk:
                        print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")
                
                chunk_cpu = audio_chunk.cpu().detach().numpy()
                
                if store_file:
                    all_chunks.append(audio_chunk)

                # Convert to mu-law format for streaming
                out, _ = (
                    ffmpeg.input('pipe:', format='f32le', acodec='pcm_f32le', ar=model.sr, channels=1)
                    .output('pipe:', format='mulaw', ar=output_sample_rate)
                    .run(input=chunk_cpu.tobytes(), capture_stdout=True, quiet=True)
                )
                yield out
                chunk_count += 1

            # Save complete audio if requested
            if store_file and all_chunks:
                final_audio = torch.cat(all_chunks, dim=-1)
                torchaudio.save(
                    os.path.join(app.config['STORAGE_FOLDER'], file_name), 
                    final_audio.unsqueeze(0), 
                    model.sr
                )
                print(f"Voice clone audio saved to {os.path.join(app.config['STORAGE_FOLDER'], file_name)}")

    return StreamingResponse(generate_voice_clone_chunks(), media_type='audio/x-mulaw')

# Route for model info
@app.get('/model/info')
async def model_info():
    """Get information about the loaded model"""
    info = {
        "model_type": "ChatterboxMultilingualTTS",
        "sample_rate": model.sr,
        "device": str(next(model.parameters()).device),
        "lora_loaded": LORA_CHECKPOINT_PATH is not None and os.path.exists(LORA_CHECKPOINT_PATH) if LORA_CHECKPOINT_PATH else False,
        "lora_path": LORA_CHECKPOINT_PATH if LORA_CHECKPOINT_PATH else None
    }
    return JSONResponse(content=info)

# Route for downloading audio files
@app.get('/download/audio')
async def download_audio(file_name: str):
    file_path = os.path.join(app.config['STORAGE_FOLDER'], file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=file_name)

# Route for deleting audio files
@app.delete('/delete/audio')
async def delete_audio(file_name: str):
    file_path = os.path.join(app.config['STORAGE_FOLDER'], file_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        return JSONResponse(content={"message": f"File {file_name} has been deleted successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# Route for serving the HTML file
@app.get('/')
async def index():
    return FileResponse(os.path.join('static', 'index.html'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=PORT)
