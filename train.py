import json
import pickle
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import mysql.connector
import logging
import argparse
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '104.243.44.92'),
    'database': os.getenv('DB_NAME', 'eddcode1_aidexchatbot'),
    'user': os.getenv('DB_USER', 'eddcode1_talkbot'),
    'password': os.getenv('DB_PASS', '(ruaN^{r)7I&'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def clean_text(text):
    """Basic text cleaning"""
    return text.lower().strip()

def get_db_connection():
    """Get database connection with retry logic"""
    for attempt in range(3):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            logger.info("Database connection established")
            return conn
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return None
            time.sleep(2)

def augment_patterns(patterns):
    """Generate variations of patterns for data augmentation"""
    augmented = patterns.copy()
    for pattern in patterns:
        # Add variations for questions
        if '?' in pattern:
            augmented.append(pattern.replace('?', ''))
            augmented.append(pattern + ' please')
            augmented.append('can you ' + pattern)
        # Add variations for greetings
        if any(word in pattern for word in ['hello', 'hi', 'hey']):
            augmented.append(pattern + ' there')
            augmented.append('hey ' + pattern)
        # Add variations for commands
        if any(word in pattern for word in ['show', 'tell', 'give']):
            augmented.append('please ' + pattern)
            augmented.append('could you ' + pattern)
    return list(set(augmented))  # Remove duplicates

def load_training_data(client_id=None):
    """Load and validate training data with class balancing"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return None
        
    try:
        query = """
            SELECT tag, patterns, responses 
            FROM chatbot_intents 
            WHERE user_token = %s OR %s IS NULL
        """ if client_id else """
            SELECT tag, patterns, responses 
            FROM chatbot_intents 
            WHERE user_token IS NULL
        """
        
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, (client_id, client_id) if client_id else ())
            rows = cursor.fetchall()
            
            if not rows:
                logger.error(f"No training data found for client: {client_id}")
                return None
                
            intents = []
            for row in rows:
                try:
                    patterns = [clean_text(p) for p in json.loads(row["patterns"]) if p.strip()]
                    responses = json.loads(row["responses"])
                    
                    if len(patterns) >= 1:  # Minimum patterns per intent
                        # Augment patterns to balance classes
                        augmented_patterns = augment_patterns(patterns)
                        
                        intents.append({
                            "tag": row["tag"],
                            "patterns": augmented_patterns,
                            "responses": responses
                        })
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for intent {row['tag']}: {e}")
                except Exception as e:
                    logger.error(f"Error processing intent {row['tag']}: {e}")
            
            logger.info(f"Loaded {len(intents)} intents after augmentation")
            return {"intents": intents} if intents else None
            
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()

def build_model(vocab_size, num_classes, max_sequence_length):
    """Create optimized LSTM model for imbalanced data"""
    model = Sequential([
        Embedding(vocab_size + 1, 128, input_length=max_sequence_length, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dropout(0.4),  # Increased dropout for regularization
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0005),  # Lower learning rate
        metrics=['accuracy']
    )
    return model

def balanced_train_test_split(X, y, test_size=0.2, min_samples=2):
    """Custom train-test split that handles imbalanced classes"""
    from collections import defaultdict
    
    # Group indices by class
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)
    
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        if len(indices) >= min_samples:
            # For classes with enough samples, do stratified split
            split_idx = max(1, int(len(indices) * (1 - test_size)))
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])
        else:
            # For small classes, put all in training
            train_indices.extend(indices)
            logger.warning(f"Class {label} has only {len(indices)} samples - using all for training")
    
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def train_model(client_id=None):
    """Main training function with class balancing"""
    try:
        # Load and prepare data
        data = load_training_data(client_id)
        if not data:
            logger.error("No valid training data available")
            return False

        # Prepare text data
        texts = []
        labels = []
        for intent in data["intents"]:
            texts.extend(intent["patterns"])
            labels.extend([intent["tag"]] * len(intent["patterns"]))

        logger.info(f"Total training samples: {len(texts)}")
        
        # Tokenize text
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        max_len = max(len(seq) for seq in sequences) if sequences else 20
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Check class distribution
        class_counts = Counter(y)
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Classes: {label_encoder.classes_}")
        
        # Use balanced train-test split
        X_train, X_val, y_train, y_val = balanced_train_test_split(
            padded, y, test_size=0.2, min_samples=2
        )
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Compute class weights for imbalanced data
        try:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            logger.info(f"Class weights: {class_weight_dict}")
        except ValueError as e:
            logger.warning(f"Could not compute class weights: {e}")
            class_weight_dict = None

        # Build and train model
        model = build_model(
            len(tokenizer.word_index),
            len(label_encoder.classes_),
            max_len
        )

        # Adjust batch size based on dataset size
        batch_size = min(32, max(8, len(X_train) // 10))
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=100,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=10, 
                    restore_best_weights=True,
                    monitor='val_loss' if len(X_val) > 0 else 'loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.2, 
                    patience=5, 
                    min_lr=1e-5,
                    monitor='val_loss' if len(X_val) > 0 else 'loss'
                )
            ],
            verbose=2
        )

        # Save model
        save_dir = os.path.join("data/models", "global" if not client_id else str(client_id))
        os.makedirs(save_dir, exist_ok=True)
        
        model.save(os.path.join(save_dir, "model.keras"))
        with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
        with open(os.path.join(save_dir, "labels.pkl"), "wb") as f:
            pickle.dump(label_encoder.classes_, f)

        logger.info("Training completed successfully")
        logger.info(f"Vocabulary size: {len(tokenizer.word_index)}")
        logger.info(f"Number of classes: {len(label_encoder.classes_)}")
        
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=str, default=None)
    args = parser.parse_args()

    if not train_model(client_id=args.client_id):
        sys.exit(1)
    sys.exit(0)