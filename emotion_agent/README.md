# Emotion Agent - Real-time Emotion Recognition System

A WebSocket-based emotion recognition system that processes audio tracks from video conference rooms in real-time and detects emotions from speech patterns.

## Architecture

The system consists of several key components:

- **Session Manager**: Manages WebSocket connections, rooms, and participant sessions
- **Emotion Manager**: Processes audio chunks and performs emotion detection
- **Audio Processor**: Handles audio format conversion, resampling, and enhancement
- **Room Management**: Tracks participants and their audio tracks per room

## Features

- ✅ Real-time audio track subscription per room
- ✅ WebSocket-based communication
- ✅ Multi-room support with isolated processing
- ✅ Audio format conversion and enhancement
- ✅ Redis integration for state persistence
- ✅ Emotion broadcasting to room participants
- ✅ Configurable emotion detection models
- ✅ Docker containerization

## Quick Start

### Using Docker (Recommended)

1. **Build the container:**
   ```bash
   docker build -t emotion-agent .
   ```

2. **Run with Redis (optional):**
   ```bash
   # Start Redis
   docker run -d --name redis -p 6379:6379 redis:alpine
   
   # Start emotion agent
   docker run -d --name emotion-agent \
     --link redis:redis \
     -p 8765:8765 \
     -e REDIS_HOST=redis \
     emotion-agent
   ```

3. **Run standalone:**
   ```bash
   docker run -d --name emotion-agent -p 8765:8765 emotion-agent
   ```

### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the service:**
   ```bash
   python run.py
   ```

## API Reference

### WebSocket Messages

#### Join Room
```json
{
  "type": "join_room",
  "user_id": "user123",
  "room_id": "room456",
  "user_info": {
    "name": "John Doe"
  }
}
```

#### Add Audio Track
```json
{
  "type": "audio_track_added",
  "track_id": "audio_track_1"
}
```

#### Send Audio Data
```json
{
  "type": "audio_data",
  "audio_data": "base64_encoded_audio_bytes",
  "sample_rate": 16000,
  "track_id": "audio_track_1"
}
```

#### Get Room Status
```json
{
  "type": "get_room_status"
}
```

### Server Responses

#### Room Joined
```json
{
  "type": "room_joined",
  "data": {
    "room_id": "room456",
    "user_id": "user123",
    "participants": ["user123", "user789"]
  }
}
```

#### Emotion Update
```json
{
  "type": "emotion_update",
  "data": {
    "user_id": "user789",
    "room_id": "room456",
    "timestamp": "2025-08-13T10:30:00Z",
    "emotion": "happiness",
    "confidence": 0.85,
    "emotion_scores": {
      "anger": 0.05,
      "disgust": 0.02,
      "fear": 0.03,
      "happiness": 0.85,
      "neutral": 0.03,
      "sadness": 0.01,
      "surprise": 0.01
    }
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server hostname |
| `REDIS_PORT` | 6379 | Redis server port |
| `WS_HOST` | 0.0.0.0 | WebSocket server host |
| `WS_PORT` | 8765 | WebSocket server port |
| `MODEL_PATH` | ./model/emotion_model.pt | Path to emotion model |
| `DEBUG` | false | Enable debug logging |

### Audio Processing Settings

- **Sample Rate**: 16kHz (configurable)
- **Supported Formats**: WAV, MP3, WebM, OGG, M4A
- **Chunk Duration**: 0.5-5.0 seconds
- **Features**: MFCC, Spectral, Chroma, ZCR

## Client Integration

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8765');

// Join room
ws.send(JSON.stringify({
  type: 'join_room',
  user_id: 'user123',
  room_id: 'conference_room_1',
  user_info: { name: 'John Doe' }
}));

// Handle emotion updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'emotion_update') {
    console.log('Emotion detected:', message.data);
  }
};

// Send audio data
function sendAudioChunk(audioBuffer) {
  const base64Audio = btoa(String.fromCharCode(...audioBuffer));
  ws.send(JSON.stringify({
    type: 'audio_data',
    audio_data: base64Audio,
    sample_rate: 16000,
    track_id: 'main_audio'
  }));
}
```

### Python Example

```python
import asyncio
from example_client import EmotionAgentClient

async def main():
    client = EmotionAgentClient("ws://localhost:8765")
    await client.connect()
    await client.join_room("user123", "room456")
    
    # Send audio data
    import numpy as np
    audio_data = np.random.randn(16000)  # 1 second of audio
    await client.send_audio_data(audio_data, 16000)
    
    await client.disconnect()

asyncio.run(main())
```

## Supported Emotions

The system detects the following emotions:
- **Anger**
- **Disgust** 
- **Fear**
- **Happiness**
- **Neutral**
- **Sadness**
- **Surprise**

## Model Training

To use your own emotion recognition model:

1. Train a PyTorch model that accepts audio features (26-40 dimensions)
2. Save the model as `model/emotion_model.pt`
3. Ensure the model outputs probabilities for 7 emotion classes
4. Restart the service

### Model Interface

Your model should implement:
```python
class EmotionModel(nn.Module):
    def forward(self, features):
        # features: (batch_size, feature_dim)
        # returns: (batch_size, 7)  # 7 emotion classes
        pass
```

## Monitoring and Logging

### Redis Keys

- `room:{room_id}:participants` - Set of active participants
- `user:{user_id}` - User session information
- `emotions:{room_id}:{user_id}` - Emotion history (last 100 results)

### Logs

The system logs to stdout with structured logging:
- Connection events
- Room join/leave events
- Audio processing results
- Error conditions

## Performance Considerations

- **Memory**: ~100MB base + model size
- **CPU**: Depends on audio processing load
- **Network**: ~50KB/s per active audio stream
- **Latency**: <200ms processing time per chunk

## Troubleshooting

### Common Issues

1. **WebSocket connection failed**
   - Check if service is running on correct port
   - Verify firewall settings

2. **Audio processing errors**
   - Ensure audio format is supported
   - Check audio chunk size (0.5-5.0 seconds)

3. **Model loading failed**
   - Verify model file exists at `MODEL_PATH`
   - Check model compatibility with PyTorch version

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
python run.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
