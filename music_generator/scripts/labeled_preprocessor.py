import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from data_preprocessor import MIDIDataProcessor

class EmotionalMIDIDataProcessor(MIDIDataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.labels_path = config['labels_path']  # Path to vgmidi.csv
        self.labels_df = None
        
    def load_data(self):
        """Load both MIDI files and their emotional labels"""
        self.labels_df = pd.read_csv(self.labels_path)
        
        # Convert midi paths to match your directory structure
        self.labels_df['midi'] = self.labels_df['midi'].apply(
            lambda x: os.path.join(self.dataset_path, x)
        )
        
        # Only load MIDI files that have labels
        self.midi_files = self.labels_df['midi'].tolist()
        return self.midi_files
    
    def preprocess_data(self, sequence_length, test_size, random_state):
        """Preprocess MIDI data and include emotional labels"""
        if not self.midi_files:
            self.load_data()
        
        all_piano_rolls = []
        all_emotions = []
        
        for idx, row in self.labels_df.iterrows():
            piano_roll = self.extract_piano_roll(row['midi'])
            if piano_roll is not None and len(piano_roll) > sequence_length:
                all_piano_rolls.append(piano_roll)
                # Create emotion vector [valence, arousal]
                emotion = np.array([row['valence'], row['arousal']])
                all_emotions.append(emotion)
        
        if not all_piano_rolls:
            return None
        
        X = []
        y = []
        emotions = []
        
        for piano_roll, emotion in zip(all_piano_rolls, all_emotions):
            for i in range(len(piano_roll) - sequence_length):
                X.append(piano_roll[i:i + sequence_length])
                y.append(piano_roll[i + sequence_length])
                emotions.append(emotion)
        
        X = np.array(X)
        y = np.array(y)
        emotions = np.array(emotions)
        
        # Split both features and emotions together
        X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
            X, y, emotions, test_size=test_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'emotions_train': e_train,
            'emotions_test': e_test
        }