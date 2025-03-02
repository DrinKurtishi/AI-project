import os
import sys
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
from sklearn.model_selection import train_test_split
pretty_midi.pretty_midi.MAX_TICK = 1e10

class MIDIDataProcessor:
    def __init__(self, config):
        """
        Initialize MIDI data processor
        
        Args:
        - config: Dictionary containing configuration parameters
        """
        self.dataset_path = config['dataset_path']
        self.fs = config['fs']
        self.midi_files = []
        self.processed_data = None

    def load_midi_files(self, max_files):
        """Load MIDI files from the dataset directory"""
        self.midi_files = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    self.midi_files.append(os.path.join(root, file))
                    if len(self.midi_files) >= max_files:
                        break
        
        print(f"Loaded {len(self.midi_files)} MIDI files")
        return self.midi_files

    def extract_piano_roll(self, midi_file):
        """Extract piano roll representation from a MIDI file"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            piano_roll = midi_data.get_piano_roll(fs=self.fs)
            piano_roll = piano_roll.T > 0
            return piano_roll.astype(np.float32)
        except Exception as e:
            print(f"Error processing {midi_file}: {str(e)}")
            return None

    def preprocess_data(self, sequence_length, test_size, random_state):
        """Preprocess MIDI data into sequences for training"""
        if not self.midi_files:
            self.load_midi_files()
        
        all_piano_rolls = []
        for midi_file in self.midi_files:
            piano_roll = self.extract_piano_roll(midi_file)
            if piano_roll is not None and len(piano_roll) > sequence_length:
                all_piano_rolls.append(piano_roll)
        
        if not all_piano_rolls:
            print("No valid MIDI features extracted!")
            return None
        
        X = []
        y = []
        
        for piano_roll in all_piano_rolls:
            for i in range(len(piano_roll) - sequence_length):
                X.append(piano_roll[i:i + sequence_length])
                y.append(piano_roll[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }