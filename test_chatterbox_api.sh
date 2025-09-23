#!/bin/bash

# ChatterBox TTS API Test Script
# Make sure the server is running before executing these commands

SERVER_URL="http://localhost:5000"

echo "=== Testing ChatterBox TTS API ==="
echo ""

# Test 1: Basic streaming audio generation
echo "1. Testing streaming audio generation (/stream/audio/speech)..."
curl -X POST "$SERVER_URL/stream/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world, this is a test of the ChatterBox TTS system.",
    "speed": 1.0,
    "output_sample_rate": 8000,
    "stream_chunk_size": 150
  }' \
  --output streaming_test.mulaw
echo "✓ Streaming audio saved to streaming_test.mulaw"
echo ""

# Test 2: Non-streaming audio generation
echo "2. Testing non-streaming audio generation (/audio/speech)..."
curl -X POST "$SERVER_URL/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a non-streaming test of ChatterBox TTS.",
    "speed": 1.0
  }' \
  --output non_streaming_test.wav
echo "✓ Non-streaming audio saved to non_streaming_test.wav"
echo ""

# Test 3: Streaming with file storage
echo "3. Testing streaming with file storage..."
curl -X POST "$SERVER_URL/stream/audio/speech?type=file&name=stored_audio.wav" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This audio will be stored on the server for later download.",
    "speed": 1.2,
    "output_sample_rate": 16000
  }' \
  --output streaming_with_storage.mulaw
echo "✓ Streaming audio with storage saved locally as streaming_with_storage.mulaw"
echo "✓ Audio should also be stored on server as stored_audio.wav"
echo ""

# Test 4: Download stored audio file
echo "4. Testing audio file download..."
curl -X GET "$SERVER_URL/download/audio?file_name=stored_audio.wav" \
  --output downloaded_audio.wav
echo "✓ Downloaded stored audio as downloaded_audio.wav"
echo ""

# Test 5: Test error handling - empty text
echo "5. Testing error handling (empty text)..."
curl -X POST "$SERVER_URL/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input": ""}' \
  -w "\nHTTP Status: %{http_code}\n"
echo ""

# Test 6: Test different speeds
echo "6. Testing different speech speeds..."
curl -X POST "$SERVER_URL/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Testing slow speech speed.",
    "speed": 0.7
  }' \
  --output slow_speech.wav
echo "✓ Slow speech saved to slow_speech.wav"

curl -X POST "$SERVER_URL/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Testing fast speech speed.",
    "speed": 1.5
  }' \
  --output fast_speech.wav
echo "✓ Fast speech saved to fast_speech.wav"
echo ""

# Test 7: Test longer text
echo "7. Testing longer text generation..."
curl -X POST "$SERVER_URL/stream/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a longer text to test the ChatterBox TTS system with more content. The system should handle longer passages of text gracefully, maintaining good quality throughout the entire generation process. This helps us verify that the streaming functionality works well with extended content.",
    "speed": 1.0,
    "output_sample_rate": 22050
  }' \
  --output long_text_test.mulaw
echo "✓ Long text audio saved to long_text_test.mulaw"
echo ""

# Test 8: Check server health (homepage)
echo "8. Testing server homepage..."
curl -X GET "$SERVER_URL/" \
  -w "\nHTTP Status: %{http_code}\n" \
  --output homepage.html
echo "✓ Homepage saved to homepage.html"
echo ""

# Test 9: Delete stored file
echo "9. Testing file deletion..."
curl -X DELETE "$SERVER_URL/delete/audio?file_name=stored_audio.wav" \
  -w "\nHTTP Status: %{http_code}\n"
echo ""

echo "=== API Testing Complete ==="
echo ""
echo "Generated files:"
echo "- streaming_test.mulaw (μ-law format)"
echo "- non_streaming_test.wav (WAV format)"
echo "- streaming_with_storage.mulaw (μ-law format)"
echo "- downloaded_audio.wav (downloaded from server)"
echo "- slow_speech.wav (0.7x speed)"
echo "- fast_speech.wav (1.5x speed)"
echo "- long_text_test.mulaw (longer text test)"
echo "- homepage.html (server homepage)"
echo ""
echo "You can play the WAV files directly with most media players."
echo "For μ-law files, you may need to convert them or use specialized tools."
