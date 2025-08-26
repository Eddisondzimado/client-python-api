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
from collections import defaultdict, Counter
import mysql.connector
import logging
import argparse
import sys
import time
import gc
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configure TensorFlow for memory optimization
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', '104.243.44.92'),
    'database': os.getenv('DB_NAME', 'eddcode1_aidexchatbot'),
    'user': os.getenv('DB_USER', 'eddcode1_talkbot'),
    'password': os.getenv('DB_PASS', '(ruaN^{r)7I&'),
    'port': int(os.getenv('DB_PORT', 3306))
}

class GoogleDriveManager:
    def __init__(self):
        try:
            logger.info("Initializing Google Drive manager...")
            
            # Try to get credentials from environment variable
            creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
            if not creds_json:
                logger.warning("GOOGLE_DRIVE_CREDENTIALS not set, Google Drive uploads will be disabled")
                self.enabled = False
                return
                
            self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
            if not self.folder_id:
                logger.warning("GOOGLE_DRIVE_FOLDER_ID not set, Google Drive uploads will be disabled")
                self.enabled = False
                return
            
            self.credentials_info = json.loads(creds_json)
            self.credentials = service_account.Credentials.from_service_account_info(
                self.credentials_info,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            self.service = build('drive', 'v3', credentials=self.credentials)
            self.enabled = True
            logger.info("Google Drive manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive: {e}")
            self.enabled = False
    
    def upload_file(self, local_path, remote_name):
        """Upload a file to Google Drive"""
        if not self.enabled:
            logger.warning("Google Drive not enabled, skipping upload")
            return None
            
        try:
            if not os.path.exists(local_path):
                logger.error(f"Local file not found: {local_path}")
                return None
            
            # Check file size and permissions
            file_size = os.path.getsize(local_path)
            logger.info(f"Uploading {remote_name} (size: {file_size} bytes)")
            
            file_metadata = {
                'name': remote_name,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(local_path)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"Uploaded {remote_name} to Google Drive with ID: {file_id}")
            return file_id
            
        except HttpError as e:
            logger.error(f"Google API error uploading {remote_name}: {e}")
            # Check specific error types
            if "insufficientFilePermissions" in str(e):
                logger.error("Service account doesn't have permission to write to this folder")
            elif "notFound" in str(e):
                logger.error("Folder not found or service account doesn't have access")
            elif "quota" in str(e).lower():
                logger.error("Google Drive quota exceeded")
            return None
        except Exception as e:
            logger.error(f"Failed to upload {remote_name}: {e}")
            return None

    def file_exists(self, remote_name):
        """Check if a file exists on Google Drive"""
        if not self.enabled:
            return False
            
        try:
            query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            files = results.get('files', [])
            return len(files) > 0
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False

# Initialize Google Drive manager
gdrive_manager = GoogleDriveManager()

def clean_text(text):
    """Basic text cleaning"""
    return text.lower().strip()

def get_db_connection():
    """Get database connection with retry logic"""
    for attempt in range(3):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            logger.warning(f"DB connection attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return None
            time.sleep(2)

def augment_patterns(patterns):
    """Generate variations of patterns for data augmentation"""
    if not patterns:
        return []
        
    augmented = set(patterns)
    for pattern in patterns:
        if not pattern or not isinstance(pattern, str):
            continue
            
        if '?' in pattern:
            augmented.add(pattern.replace('?', ''))
            augmented.add(pattern + ' please')
            augmented.add('can you ' + pattern)
        if any(word in pattern for word in ['hello', 'hi', 'hey']):
            augmented.add(pattern + ' there')
            augmented.add('hey ' + pattern)
        if any(word in pattern for word in ['show', 'tell', 'give']):
            augmented.add('please ' + pattern)
            augmented.add('could you ' + pattern)
    return list(augmented)

def load_training_data(client_id=None):
    """Load and validate training data"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return None
        
    try:
        query = """
            SELECT tag, patterns, responses 
            FROM chatbot_intents 
            WHERE (user_token = %s OR %s IS NULL) AND patterns IS NOT NULL AND responses IS NOT NULL
        """ if client_id else """
            SELECT tag, patterns, responses 
            FROM chatbot_intents 
            WHERE user_token IS NULL AND patterns IS NOT NULL AND responses IS NOT NULL
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
                    if not row["patterns"] or not row["responses"]:
                        continue
                        
                    patterns = [clean_text(p) for p in json.loads(row["patterns"]) if p and p.strip()]
                    responses = json.loads(row["responses"])
                    
                    if len(patterns) >= 1:
                        augmented_patterns = augment_patterns(patterns)
                        intents.append({
                            "tag": row["tag"],
                            "patterns": augmented_patterns,
                            "responses": responses
                        })
                    else:
                        logger.warning(f"Skipping intent '{row['tag']}' with no valid patterns")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for intent {row['tag']}: {e}")
                except Exception as e:
                    logger.error(f"Error processing intent {row['tag']}: {e}")
            
            logger.info(f"Loaded {len(intents)} intents with {sum(len(i['patterns']) for i in intents)} total patterns")
            return {"intents": intents} if intents else None
            
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None
    finally:
        if conn:
            conn.close()

def build_optimized_model(vocab_size, num_classes, max_sequence_length):
    """Create memory-optimized LSTM model"""
    model = Sequential([
        Embedding(vocab_size + 1, 64, input_length=max_sequence_length, mask_zero=True),
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(16)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    return model

def balanced_train_test_split(X, y, test_size=0.2, min_samples=2):
    """Custom train-test split for imbalanced classes"""
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)
    
    train_indices, val_indices = [], []
    
    for label, indices in class_indices.items():
        if len(indices) >= min_samples:
            split_idx = max(1, int(len(indices) * (1 - test_size)))
            train_indices.extend(indices[:split_idx])
            val_indices.extend(indices[split_idx:])
        else:
            train_indices.extend(indices)
            logger.warning(f"Class {label} has only {len(indices)} samples")
    
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def upload_to_google_drive(client_id, model_path, tokenizer_path, labels_path):
    """Upload model files to Google Drive"""
    if not gdrive_manager.enabled:
        logger.warning("Google Drive not enabled, skipping upload")
        return False
        
    try:
        prefix = "global" if not client_id else f"client_{client_id}"
        
        # Check files exist
        for path in [model_path, tokenizer_path, labels_path]:
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return False
        
        # Upload files
        upload_results = []
        files_to_upload = [
            (model_path, f"{prefix}_model.keras"),
            (tokenizer_path, f"{prefix}_tokenizer.pkl"),
            (labels_path, f"{prefix}_labels.pkl")
        ]
        
        for local_path, remote_name in files_to_upload:
            # Check if file already exists on Google Drive
            if gdrive_manager.file_exists(remote_name):
                logger.info(f"File {remote_name} already exists on Google Drive, skipping upload")
                upload_results.append(True)
            else:
                file_id = gdrive_manager.upload_file(local_path, remote_name)
                upload_results.append(file_id is not None)
        
        success = all(upload_results)
        if success:
            logger.info("All files processed on Google Drive")
        else:
            logger.error("Some files failed to upload")
        
        return success
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False

def train_model(client_id=None):
    """Main training function with memory optimization"""
    try:
        # Load data
        data = load_training_data(client_id)
        if not data:
            logger.error("No training data available")
            return False

        # Prepare text data
        texts, labels = [], []
        for intent in data["intents"]:
            texts.extend(intent["patterns"])
            labels.extend([intent["tag"]] * len(intent["patterns"]))

        logger.info(f"Training samples: {len(texts)}")
        
        if len(texts) == 0:
            logger.error("No valid training texts found")
            return False
            
        # Tokenize text
        tokenizer = Tokenizer(oov_token="<OOV>", filters='')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        max_len = max(len(seq) for seq in sequences) if sequences else 20
        max_len = min(max_len, 100)  # Cap sequence length to prevent memory issues
        padded = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = balanced_train_test_split(padded, y)
        
        # Compute class weights
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            logger.info(f"Class weights computed: {class_weight_dict}")
        except ValueError as e:
            logger.warning(f"Could not compute class weights: {e}")
            class_weight_dict = None

        # Build and train model
        model = build_optimized_model(len(tokenizer.word_index), len(label_encoder.classes_), max_len)

        # Adjust batch size for memory optimization
        batch_size = min(16, max(4, len(X_train) // 20))
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train))
        
        logger.info(f"Training configuration: batch_size={batch_size}, samples={len(X_train)}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=50,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=4)
            ],
            verbose=1
        )

        # Save model
        save_dir = os.path.join("data/models", "global" if not client_id else str(client_id))
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "model.keras")
        tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
        labels_path = os.path.join(save_dir, "labels.pkl")
        
        # Save files
        model.save(model_path, save_format='tf')
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f, protocol=4)
        with open(labels_path, "wb") as f:
            pickle.dump(label_encoder.classes_, f, protocol=4)

        # Upload to Google Drive
        upload_success = upload_to_google_drive(client_id, model_path, tokenizer_path, labels_path)
        
        if not upload_success:
            logger.warning("Google Drive upload failed, but local files saved")

        # Clean up memory
        del model, tokenizer, label_encoder, X_train, X_val, y_train, y_val
        gc.collect()

        logger.info("Training completed successfully")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train chatbot model")
    parser.add_argument("--client-id", type=str, default=None, help="Client ID for specific training")
    args = parser.parse_args()

    logger.info(f"Starting training for client: {args.client_id or 'global'}")
    
    if not train_model(client_id=args.client_id):
        logger.error("Training failed")
        sys.exit(1)
    
    logger.info("Training completed successfully")
    sys.exit(0)