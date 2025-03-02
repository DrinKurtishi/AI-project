# Emotional Music Generation Using Deep Learning

## Project Overview

### Introduction
This project implements an advanced emotional music generation system using deep learning techniques, specifically leveraging LSTM networks and emotional conditioning. The system generates original musical pieces while incorporating emotional parameters (valence and arousal) to control the mood and style of the generated music.

## Core Features
- Emotion-controlled music generation through valence and arousal parameters
- Multi-instrument support with specialized soundfonts
- Dual output format support (MIDI and MP3)
- Customizable generation parameters for fine-tuned control
- Real-time music synthesis capabilities
- Labeled dataset with emotional annotations

## Technical Architecture

### Neural Network Architecture
#### LSTM Implementation
The model uses a sophisticated dual-LSTM architecture:

**First LSTM Layer (1024 units):**
- Processes basic musical patterns
- Identifies common note combinations
- Captures rhythmic patterns
- Analyzes note transitions
- Returns sequences for deeper processing

**Second LSTM Layer (512 units):**
- Processes higher-level patterns
- Identifies melodic phrases
- Recognizes chord progressions
- Understands musical structure
- Provides consolidated musical understanding

**Dropout Layers (0.3 rate):**
- Prevents overfitting during training
- Randomly deactivates 30% of neurons
- Forces robust pattern learning
- Ensures generalization capability

### Emotion Processing Pipeline
#### Emotion Input Layer
- Accepts valence and arousal values
- Valence range: `-1.0` (sad) to `1.0` (happy)
- Arousal range: `-1.0` (calm) to `1.0` (energetic)

#### Emotion Embedding
- Initial dense layer: `64` units with ReLU activation
- Secondary dense layer: `128` units with ReLU activation
- Creates rich emotional feature representation

#### Feature Integration
- Repeats emotion embedding to match sequence length
- Concatenates with musical features
- Ensures emotion-aware music generation

## Data Processing System

### MIDI Data Processing
#### Piano Roll Conversion
- Resolution: `16th note (fs=16)`
- Full MIDI note range: `0-127`
- Binary note representation: `on/off`

#### Sequence Formation
- Sequence length: `64` timesteps
- Sliding window approach for training data
- Overlapping sequences for continuity

### Training Data Structure
#### `X_train` Format
Shape: `(num_sequences, 64, 128)`
```plaintext
Timestep 1: [1,0,1,0,0,1,0,0,...] # 128 values
Timestep 2: [0,1,1,0,0,1,0,0,...]
...
Timestep 64: [0,1,0,0,1,0,1,0,...]
```

#### `y_train` Format
Shape: `(num_sequences, 128)`
```plaintext
y_train[0] = [1,1,0,0,...] # Next timestep prediction
```

## Generation Process

### Music Generation Parameters
#### Length Control
- **Short Generation (length=32):**
  - ~2 seconds of music
  - Suitable for motifs and patterns
  - High coherence maintenance
- **Medium Generation (length=300):**
  - ~19 seconds of music
  - Ideal for complete musical phrases
  - Balanced structure maintenance
- **Long Generation (length=1000):**
  - ~62 seconds of music
  - Full musical compositions
  - Requires careful coherence management

#### Temperature Control
Temperature affects note selection randomness:
- **Low Temperature (0.5)**: More structured, adheres to learned patterns
- **Normal Temperature (1.0)**: Balanced creativity, ideal for most cases
- **High Temperature (1.5)**: More experimental, increased randomness

### Output Processing
#### MIDI Generation
- Handles note-on and note-off events
- Manages velocity variations
- Implements style-specific parameters

#### Style Parameters
- Tempo control (BPM)
- Velocity range customization
- Articulation management
- Rhythmic precision control

### Audio Conversion
#### Soundfont Implementation
Multiple instrument profiles:
- Contra
- Nintendo
- Violin
- Piano
- Custom soundfont support

#### MP3 Conversion Pipeline
- WAV intermediate generation
- FFmpeg optimization
- Quality control checks

## Implementation Details

### Training Configuration
```python
config = {
    'sequence_length': 64,
    'test_size': 0.2,
    'fs': 16,
    'note_range': 128,
    'lstm_units': (1024, 512),
    'dense_units': 256,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'epochs': 1,
    'batch_size': 32,
    'patience': 5,
    'generation_length': 300
}
```

### Generation Command Examples
```sh
# Happy, Calm Composition
generate_music.py \
  --valence 1.0 \
  --arousal -1.0 \
  --tempo 120 \
  --velocity_min 64 \
  --velocity_max 127 \
  --generation_length 300 \
  --soundfont contra \
  --output_name happy_calm_song
```

## Future Development

### Planned Enhancements
#### Advanced Emotional Parameter Integration
- Multiple emotion dimensions
- Dynamic emotion transitions
- Contextual emotion processing

#### Extended Musical Capabilities
- Multi-track generation
- Complex harmony support
- Dynamic structure control

#### Technical Improvements
- Real-time generation optimization
- Advanced model architectures
- Enhanced training efficiency

## Technical Requirements

### Software Dependencies
- Python 3.10
- TensorFlow
- Mido for MIDI processing
- FluidSynth
- FFmpeg

### Hardware Requirements
- Minimum 8GB RAM
- CUDA-capable GPU (recommended)

