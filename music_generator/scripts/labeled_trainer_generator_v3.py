# In your model_trainer_generator_v3.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Concatenate, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message
import sys
from labeled_preprocessor import EmotionalMIDIDataProcessor
from midi2audio import FluidSynth
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pretty_midi")
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

class EmotionalMusicGenerationModel:
    def __init__(self, config, input_shape=None):
        self.note_range = config['note_range']
        self.model_path = config.get('model_save_path')
        
        if self.model_path and input_shape is None:
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            self.model = self.build_model(input_shape, config)

    def build_model(self, input_shape, config):
        """Build model with emotional conditioning"""
        # Music input
        music_input = Input(shape=input_shape)
        
        # Emotion input (valence, arousal)
        emotion_input = Input(shape=(2,))
        
        # Emotion embedding
        emotion_embedding = Dense(64, activation='relu')(emotion_input)
        emotion_embedding = Dense(128, activation='relu')(emotion_embedding)
        
        # Repeat emotion embedding to match sequence length
        emotion_repeated = tf.keras.layers.RepeatVector(input_shape[0])(emotion_embedding)
        
        # Concatenate music and emotion features
        combined_input = Concatenate()([music_input, emotion_repeated])
        
        # LSTM layers
        x = LSTM(config['lstm_units'][0], return_sequences=True)(combined_input)
        x = Dropout(config['dropout_rate'])(x)
        x = LSTM(config['lstm_units'][1])(x)
        x = Dropout(config['dropout_rate'])(x)
        
        # Dense layers
        x = Dense(config['dense_units'], activation='relu')(x)
        output = Dense(self.note_range, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=[music_input, emotion_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_midi(self, piano_roll, config, style_params):
        """Convert piano roll to MIDI file with style-specific parameters"""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Use style-specific tempo
        tempo_us = mido.bpm2tempo(style_params['tempo'])
        track.append(mido.MetaMessage('set_tempo', tempo=tempo_us))
        
        ticks_per_beat = midi.ticks_per_beat
        ticks_per_step = ticks_per_beat / 4
        
        current_notes = set()
        
        for step, notes in enumerate(piano_roll):
            new_notes = set(np.where(notes > 0)[0])
            
            # Handle note-offs
            for note in current_notes - new_notes:
                track.append(Message('note_off', note=int(note), velocity=64, time=0))
            
            # Handle note-ons with style-specific velocity
            for note in new_notes - current_notes:
                velocity = np.random.randint(
                    style_params['velocity_range'][0],
                    style_params['velocity_range'][1]
                )
                track.append(Message('note_on', note=int(note),
                                   velocity=velocity, time=int(ticks_per_step)))
            
            current_notes = new_notes
        
        # Final note-offs
        for note in current_notes:
            track.append(Message('note_off', note=int(note), velocity=64, time=0))
        
        midi.save(config['output_path'])
    
    def save_midi_no_styles(self, piano_roll, config):
        """Convert piano roll to MIDI file with style-specific parameters"""
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        ticks_per_beat = midi.ticks_per_beat
        ticks_per_step = ticks_per_beat / 4
        
        current_notes = set()
        
        for step, notes in enumerate(piano_roll):
            new_notes = set(np.where(notes > 0)[0])
            
            # Handle note-offs
            for note in current_notes - new_notes:
                track.append(Message('note_off', note=int(note), velocity=64, time=0))
            
            # Handle note-ons with style-specific velocity
            for note in new_notes - current_notes:
                track.append(Message('note_on', note=int(note), time=int(ticks_per_step)))
            
            current_notes = new_notes
        
        # Final note-offs
        for note in current_notes:
            track.append(Message('note_off', note=int(note), velocity=64, time=0))
        
        midi.save(config['output_path'])

    def train(self, X_train, y_train, X_test, y_test, emotions_train, emotions_test, config):
        """Train the model with early stopping and checkpointing"""
        callbacks = []
        
        if self.model_path:
            checkpoint = ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True
            )
            callbacks.append(checkpoint)

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True
        )
        callbacks.append(early_stop)

        history = self.model.fit(
            [X_train, emotions_train], y_train,
            validation_data=([X_test, emotions_test], y_test),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        if self.model_path:
            self.model.save(self.model_path)
        return history

    def generate_music(self, seed_sequence, emotion_values, config):
        """Generate music with specified emotion values"""
        current_sequence = seed_sequence.copy()
        generated_sequence = []
        
        # Prepare emotion input
        emotion_input = np.array([[emotion_values['valence'], emotion_values['arousal']]])
        
        for _ in range(config['generation_length']):
            pred_input = current_sequence.reshape(1, current_sequence.shape[0], self.note_range)
            predicted_notes = self.model.predict([pred_input, emotion_input], verbose=0)[0]
            
            # Apply temperature scaling for variety
            predicted_notes = np.random.binomial(1, predicted_notes)
            
            generated_sequence.append(predicted_notes)
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = predicted_notes

        return np.array(generated_sequence)

class MIDItoMP3Converter:
    def __init__(self, soundfonts_dir):
        """Initialize converter with soundfonts directory"""
        self.soundfonts_dir = soundfonts_dir
        self.soundfonts = {
            'contra': 'contra.sf2',
            'nintendo': 'nintendo.sf2',
            'violin': 'violin.sf2',
            'piano': 'piano.sf2'
        }

    def verify_midi(self, midi_path):
        """Verify MIDI file is valid and has content"""
        try:
            midi_file = MidiFile(midi_path)
            total_ticks = sum(msg.time for msg in midi_file)
            total_messages = sum(1 for track in midi_file.tracks for msg in track)
            
            if total_ticks == 0 or total_messages == 0:
                raise ValueError("MIDI file appears to be empty")
                
            return True
        except Exception as e:
            return False
    
    def convert(self, midi_path, output_path, instrument):
        """Convert MIDI to MP3 using specified soundfont"""
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
            
        if not self.verify_midi(midi_path):
            raise ValueError("Invalid MIDI file")
            
        if instrument not in self.soundfonts:
            raise ValueError(f"Unknown instrument: {instrument}")
            
        soundfont_path = os.path.join(self.soundfonts_dir, self.soundfonts[instrument])
        
        if not os.path.exists(soundfont_path):
            raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")
        
        try:
            # Convert to WAV first
            wav_path = output_path.replace('.mp3', '.wav')
            
            # Create a more robust output suppression context
            with open(os.devnull, 'w') as devnull:
                # Save original file descriptors
                original_stdout_fd = os.dup(1)
                original_stderr_fd = os.dup(2)
                
                try:
                    # Redirect stdout and stderr to devnull at the file descriptor level
                    os.dup2(devnull.fileno(), 1)
                    os.dup2(devnull.fileno(), 2)
                    
                    fs = FluidSynth(sound_font=soundfont_path)
                    fs.midi_to_audio(midi_path, wav_path)
                finally:
                    # Restore original file descriptors
                    os.dup2(original_stdout_fd, 1)
                    os.dup2(original_stderr_fd, 2)
                    # Close the saved file descriptors
                    os.close(original_stdout_fd)
                    os.close(original_stderr_fd)
            
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                raise RuntimeError("WAV file creation failed")
                
            # Convert WAV to MP3 using ffmpeg
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-y',
                '-loglevel', 'quiet',
                '-i', wav_path,
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',
                output_path
            ], text=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
            
            if result.returncode != 0:
                raise RuntimeError("FFmpeg conversion failed")
                
            # Clean up temporary WAV file
            os.remove(wav_path)
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("MP3 file creation failed")
            
        except Exception as e:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            raise


def get_unique_filename(base_dir, desired_name):
    """
    Check if filename exists and handle duplicates.
    Returns a unique filename by appending numbers if necessary.
    """
    # Ensure the filename ends with .mid
    if not desired_name.endswith('.mid'):
        desired_name += '.mid'
    
    # Get the full path
    file_path = Path(base_dir) / desired_name
    
    # If file doesn't exist, return the original name
    if not file_path.exists():
        return desired_name
    
    # If file exists, add numbers until we find a unique name
    base_name = desired_name[:-4]  # Remove .mid extension
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}.mid"
        new_path = Path(base_dir) / new_name
        if not new_path.exists():
            return new_name
        counter += 1

def get_file_paths(config, desired_name):
    """
    Generate paths for MIDI and MP3 files based on the desired name.
    """
    midi_dir = Path(config['output_path']).parent
    mp3_dir = Path(config['mp3_output_path']).parent
    
    # Ensure directories exist
    midi_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove any existing extension the user might have added
    desired_name = Path(desired_name).stem
    
    # Get unique filename
    midi_filename = get_unique_filename(midi_dir, desired_name)
    
    # Generate the full paths
    midi_path = str(midi_dir / midi_filename)
    mp3_path = str(mp3_dir / midi_filename.replace('.mid', '.mp3'))
    
    return midi_path, mp3_path

# Modified main function
def main():
    warnings.filterwarnings('ignore')
    config = {
        # Your existing paths
        'dataset_path': '/Users/drinkurtishi/Desktop/AI-project/music_generator/data/8bit-dataset',
        'model_save_path': '/Users/drinkurtishi/Desktop/AI-project/music_generator/models/8bit_labeled_model_v1.h5',
        'output_path': '/Users/drinkurtishi/Desktop/AI-project/music_generator/data/generated_music/8bit-labeled-happycalm.mid',
        'labels_path': '/Users/drinkurtishi/Desktop/AI-project/music_generator/data/vgmidi_labelled.csv',
        'mp3_output_path': '/Users/drinkurtishi/Desktop/AI-project/music_generator/data/generated_music_mp3/8bit-labeled-happycalm.mp3',
        'soundfonts_dir': '/Users/drinkurtishi/Desktop/AI-project/music_generator/data/soundfonts',
        
        # Your existing parameters
        'sequence_length': 64,
        'test_size': 0.2,
        'fs': 16,
        'random_state': 42,
        'note_range': 128,
        'lstm_units': (1024, 512),
        'dense_units': 256,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 1,
        'batch_size': 32,
        'patience': 5,
        'generation_length': 300,
        'instrument': 'contra',
    }

    action = sys.argv[1] if len(sys.argv) > 1 else 'train'

    # Use emotional processor instead of original
    processor = EmotionalMIDIDataProcessor(config)
    processed_data = processor.preprocess_data(
        sequence_length=config['sequence_length'],
        test_size=config['test_size'],
        random_state=config['random_state']
    )

    if action == 'train':
        model = EmotionalMusicGenerationModel(
            config=config,
            input_shape=(processed_data['X_train'].shape[1], config['note_range'])
        )

        model.train(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_test'],
            processed_data['y_test'],
            processed_data['emotions_train'],
            processed_data['emotions_test'],
            config
        )

    elif action == 'generate':
        parser = argparse.ArgumentParser()
        parser.add_argument('--valence', type=float, default=1.0,
                            help='Valence value (-1.0 for sad to 1.0 for happy)')
        parser.add_argument('--arousal', type=float, default=-1.0,
                            help='Arousal value (-1.0 for calm to 1.0 for energetic)')
        parser.add_argument('--tempo', type=int, default=120,
                            help='Tempo in BPM')
        parser.add_argument('--velocity_min', type=int, default=64,
                            help='Minimum velocity for notes')
        parser.add_argument('--velocity_max', type=int, default=127,
                            help='Maximum velocity for notes')
        parser.add_argument('--generation_length', type=int, default=config['generation_length'],
                            help='Length of the generated music sequence')
        parser.add_argument('--soundfont', type=str, default=config['instrument'],
                            help='Soundfont to use for MP3 conversion (e.g., "contra", "nintendo")')
        parser.add_argument('--output_name', type=str, required=True,
                            help='Desired output filename (without extension)')
        args = parser.parse_args(sys.argv[2:])

        midi_path, mp3_path = get_file_paths(config, args.output_name)

        config['generation_length'] = args.generation_length
        config['instrument'] = args.soundfont
        config['output_path'] = midi_path
        config['mp3_output_path'] = mp3_path

        seed_index = np.random.randint(0, processed_data['X_test'].shape[0])
        seed_sequence = processed_data['X_test'][seed_index]

        emotion_values = {
            'valence': args.valence,
            'arousal': args.arousal
        }

        style_params = {
            'tempo': args.tempo,
            'velocity_range': (args.velocity_min, args.velocity_max)
        }

        model = EmotionalMusicGenerationModel(config=config)
        generated_music = model.generate_music(seed_sequence, emotion_values, config)
        
        model.save_midi(generated_music, config, style_params)

        # Convert to MP3
        try:
            converter = MIDItoMP3Converter(config['soundfonts_dir'])
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            null_fh = open(os.devnull, 'w')
            sys.stdout = null_fh
            sys.stderr = null_fh
            converter.convert(
                 config['output_path'],
                 config['mp3_output_path'],
                 config['instrument']
             )
            response = {
                'success': True,
                'message': 'Music generated successfully',
                'midi_path': config['output_path'],
                'mp3_path': config['mp3_output_path']
            }   
        except Exception as e:
            response = {
                'success': False,
                'message': str(e)
            }
        finally:
                # Restore original stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                null_fh.close()

    print(json.dumps(response))

if __name__ == '__main__':
    main()