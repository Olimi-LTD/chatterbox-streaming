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
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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

# Load environment variables
load_dotenv()

PORT = int(os.getenv('PORT', 5000))

# Initialize the FastAPI app
app = FastAPI()
app.config = {"STORAGE_FOLDER": "./storage"}

cwd = os.getcwd()

# Load the ChatterBox model
print("Loading ChatterBox model...")
LORA_CHECKPOINT_PATH = "/home/incode/projects/chatterbox/chatterbox-streaming/checkpoint_epoch24_step14880.pt"

model = load_lora_model(LORA_CHECKPOINT_PATH, device="cuda")

# Semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(10)  # Adjust based on GPU capacity

# Route to synthesize speech
@app.post('/stream/audio/speech')
async def synthesize(request: Request):
    data = await request.json()
    input_text = data.get('input', '')
    speed = data.get('speed', 1.0)
    output_sample_rate = data.get('output_sample_rate', 8000)
    stream_chunk_size = data.get('stream_chunk_size', 150)

    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")

    # Check if we need to store the audio in a file
    store_file = request.query_params.get('type') == 'file'
    file_name = request.query_params.get('name', 'output.wav') if store_file else None

    print('Received stream data:', data)

    async def generate_chunks():
        async with semaphore:
            t0 = time.time()
            
            all_chunks = []
            chunk_count = 0

            # Use ChatterBox streaming generation
            for audio_chunk, metrics in model.generate_stream(input_text):
                if chunk_count == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                
                print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if metrics.rtf else f"Chunk {metrics.chunk_count}")

                # Move tensor to CPU and convert to numpy array
                chunk_cpu = audio_chunk.cpu().detach().numpy()
                print(f"Chunk shape: {chunk_cpu.shape}, {chunk_cpu}")

                if store_file:
                    all_chunks.append(audio_chunk)

                # Use ffmpeg-python to convert numpy array to mu-law format
                out, _ = (
                    ffmpeg.input('pipe:', format='f32le', acodec='pcm_f32le', ar=model.sr, channels=1)
                    .output('pipe:', format='mulaw', ar=output_sample_rate)
                    .run(input=chunk_cpu.tobytes(), capture_stdout=True, quiet=True)
                )
                yield out
                chunk_count += 1

            # Save to file if required
            if store_file and all_chunks:
                combined_audio = torch.cat(all_chunks, dim=-1)
                torchaudio.save(
                    os.path.join(app.config['STORAGE_FOLDER'], file_name), 
                    combined_audio.unsqueeze(0), 
                    model.sr
                )
                print(f"Audio saved to {os.path.join(app.config['STORAGE_FOLDER'], file_name)}")

    return StreamingResponse(generate_chunks(), media_type='audio/x-mulaw')

# Route for non-streaming inference
@app.post('/audio/speech')
async def generate_audio(request: Request):
    data = await request.json()
    input_text = data.get('input', '')
    speed = data.get('speed', 1.0)

    if not input_text:
        raise HTTPException(status_code=400, detail="No text provided")

    async def generate_audio_stream():
        # Collect all chunks for non-streaming response
        audio_chunks = []
        for audio_chunk, metrics in model.generate_stream(input_text):
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
