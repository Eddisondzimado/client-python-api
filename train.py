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
import base64

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

# Try to import Google Cloud Storage
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
    logger.info("Google Cloud Storage libraries available")
except ImportError:
    logger.warning("Google Cloud Storage libraries not available")
    GCS_AVAILABLE = False

class GoogleCloudStorageManager:
    def __init__(self):
        self.enabled = False
        self.bucket = None
        self.client = None
        self.bucket_name = None
        
        try:
            # Get bucket name from environment
            self.bucket_name = os.environ.get('GCS_BUCKET_NAME')
            if not self.bucket_name:
                logger.warning("GCS_BUCKET_NAME not set, Google Cloud Storage disabled")
                return
            
            logger.info(f"Initializing GCS with bucket: {self.bucket_name}")
            
            # Check if we have base64 encoded credentials
            creds_b64 = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if creds_b64:
                try:
                    # Decode base64 and parse JSON
                    creds_json = base64.b64decode(creds_b64).decode('utf-8')
                    credentials_info = json.loads(creds_json)
                    credentials = service_account.Credentials.from_service_account_info(credentials_info)
                    self.client = storage.Client(credentials=credentials)
                    logger.info("Using provided service account credentials from base64")
                except Exception as e:
                    logger.error(f"Failed to parse credentials: {e}")
                    logger.error(f"Credentials length: {len(creds_b64)}")
                    return
            else:
                # Use default credentials (when running on Google Cloud)
                logger.info("Using default application credentials")
                self.client = storage.Client()
            
            # Get bucket
            try:
                self.bucket = self.client.bucket(self.bucket_name)
                if not self.bucket.exists():
                    logger.error(f"Bucket {self.bucket_name} does not exist")
                    # Try to create the bucket if it doesn't exist
                    try:
                        logger.info(f"Attempting to create bucket: {self.bucket_name}")
                        self.bucket.create()
                        logger.info(f"Bucket {self.bucket_name} created successfully")
                    except Exception as create_error:
                        logger.error(f"Failed to create bucket: {create_error}")
                        return
                else:
                    logger.info(f"Bucket {self.bucket_name} exists and accessible")
                    
                    # Test bucket permissions by listing a few files
                    try:
                        blobs = list(self.bucket.list_blobs(max_results=2))
                        logger.info(f"Bucket list test successful, found {len(blobs)} items")
                    except Exception as list_error:
                        logger.error(f"Bucket list test failed: {list_error}")
                        return
                    
            except Exception as e:
                logger.error(f"Failed to access bucket {self.bucket_name}: {e}")
                return
            
            self.enabled = True
            logger.info(f"Google Cloud Storage manager initialized successfully with bucket: {self.bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.enabled = False
    
    def upload_file(self, local_path, remote_path):
        """Upload a file to Google Cloud Storage"""
        if not self.enabled:
            logger.warning("Google Cloud Storage not enabled, skipping upload")
            return False
            
        try:
            if not os.path.exists(local_path):
                logger.error(f"Local file not found: {local_path}")
                return False
            
            file_size = os.path.getsize(local_path)
            logger.info(f"Uploading {local_path} ({file_size} bytes) to gs://{self.bucket_name}/{remote_path}")
            
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            
            # Verify upload was successful
            if blob.exists():
                logger.info(f"Successfully uploaded to gs://{self.bucket_name}/{remote_path}")
                return True
            else:
                logger.error(f"Upload verification failed: gs://{self.bucket_name}/{remote_path} does not exist")
                return False
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to GCS: {e}")
            import traceback
            logger.error(f"Upload traceback: {traceback.format_exc()}")
            return False
    
    def save_model(self, client_id, model, tokenizer, label_encoder_classes):
        """Save model components to GCS"""
        if not self.enabled:
            logger.warning("Google Cloud Storage not enabled, skipping GCS save")
            return False
            
        try:
            prefix = "global" if not client_id else f"client_{client_id}"
            logger.info(f"Saving model to GCS with prefix: {prefix}")
            
            # Create temporary directory for upload
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model.keras")
                tokenizer_path = os.path.join(temp_dir, "tokenizer.pkl")
                labels_path = os.path.join(temp_dir, "labels.pkl")
                
                # Save files locally first
                logger.info("Saving model components to temporary files...")
                try:
                    model.save(model_path)
                    logger.info(f"Model saved locally: {model_path}")
                    
                    with open(tokenizer_path, "wb") as f:
                        pickle.dump(tokenizer, f)
                    logger.info(f"Tokenizer saved locally: {tokenizer_path}")
                    
                    with open(labels_path, "wb") as f:
                        pickle.dump(label_encoder_classes, f)
                    logger.info(f"Labels saved locally: {labels_path}")
                except Exception as save_error:
                    logger.error(f"Failed to save model components locally: {save_error}")
                    return False
                
                # Upload to GCS
                success = True
                logger.info("Uploading model components to GCS...")
                
                success &= self.upload_file(model_path, f"{prefix}/model.keras")
                success &= self.upload_file(tokenizer_path, f"{prefix}/tokenizer.pkl")
                success &= self.upload_file(labels_path, f"{prefix}/labels.pkl")
                
                if success:
                    logger.info(f"Model successfully saved to Google Cloud Storage: {prefix}")
                    
                    # Verify all files exist in GCS
                    try:
                        blob_model = self.bucket.blob(f"{prefix}/model.keras")
                        blob_tokenizer = self.bucket.blob(f"{prefix}/tokenizer.pkl")
                        blob_labels = self.bucket.blob(f"{prefix}/labels.pkl")
                        
                        logger.info(f"GCS Verification - Model exists: {blob_model.exists()}")
                        logger.info(f"GCS Verification - Tokenizer exists: {blob_tokenizer.exists()}")
                        logger.info(f"GCS Verification - Labels exist: {blob_labels.exists()}")
                        
                    except Exception as verify_error:
                        logger.error(f"GCS verification failed: {verify_error}")
                        
                else:
                    logger.error("Failed to save some model components to Google Cloud Storage")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to save model to GCS: {e}")
            import traceback
            logger.error(f"Save model traceback: {traceback.format_exc()}")
            return False

# Initialize GCS manager
gcs_manager = GoogleCloudStorageManager() if GCS_AVAILABLE else None

def clean_text(text):
    """Basic text cleaning"""
    if not text or not isinstance(text, str):
        return ""
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
    if not patterns:
        return []
        
    augmented = set(patterns)
    for pattern in patterns:
        if not pattern or not isinstance(pattern, str):
            continue
            
        # Add variations for questions
        if '?' in pattern:
            augmented.add(pattern.replace('?', ''))
            augmented.add(pattern + ' please')
            augmented.add('can you ' + pattern)
        # Add variations for greetings
        if any(word in pattern for word in ['hello', 'hi', 'hey']):
            augmented.add(pattern + ' there')
            augmented.add('hey ' + pattern)
        # Add variations for commands
        if any(word in pattern for word in ['show', 'tell', 'give']):
            augmented.add('please ' + pattern)
            augmented.add('could you ' + pattern)
    
    return list(augmented)

def load_training_data(client_id=None):
    """Load and validate training data with class balancing"""
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return None
        
    try:
        if client_id:
            query = """
                SELECT tag, patterns, responses 
                FROM chatbot_intents 
                WHERE user_token = %s AND patterns IS NOT NULL AND responses IS NOT NULL
            """
            params = (client_id,)
        else:
            query = """
                SELECT tag, patterns, responses 
                FROM chatbot_intents 
                WHERE user_token IS NULL AND patterns IS NOT NULL AND responses IS NOT NULL
            """
            params = ()
        
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, params)
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
                    
                    if len(patterns) >= 1:  # Minimum patterns per intent
                        # Augment patterns to balance classes
                        augmented_patterns = augment_patterns(patterns)
                        
                        intents.append({
                            "tag": row["tag"],
                            "patterns": augmented_patterns,
                            "responses": responses
                        })
                        logger.debug(f"Added intent: {row['tag']} with {len(augmented_patterns)} patterns")
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
        if conn and conn.is_connected():
            conn.close()

def build_model(vocab_size, num_classes, max_sequence_length):
    """Create optimized LSTM model for imbalanced data"""
    model = Sequential([
        Embedding(vocab_size + 1, 64, input_length=max_sequence_length, mask_zero=True),  # Reduced from 128
        Bidirectional(LSTM(32, return_sequences=True)),  # Reduced from 64
        Dropout(0.3),  # Reduced from 0.5
        Bidirectional(LSTM(16)),  # Reduced from 32
        Dropout(0.2),  # Reduced from 0.4
        Dense(32, activation='relu'),  # Reduced from 64
        Dropout(0.2),  # Reduced from 0.3
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    return model

def balanced_train_test_split(X, y, test_size=0.2, min_samples=2):
    """Custom train-test split that handles imbalanced classes"""
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
        
        if len(texts) == 0:
            logger.error("No valid training texts found")
            return False
            
        # Tokenize text
        tokenizer = Tokenizer(oov_token="<OOV>", filters='')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        max_len = max(len(seq) for seq in sequences) if sequences else 20
        max_len = min(max_len, 100)  # Cap sequence length
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
        
        if len(X_train) == 0:
            logger.error("No training samples after split")
            return False
        
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
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train))
        
        logger.info(f"Training configuration: batch_size={batch_size}, epochs=100")
        
        history = model.fit(
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
            verbose=1
        )

        # Save model locally
        save_dir = os.path.join("data/models", "global" if not client_id else str(client_id))
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "model.keras")
        tokenizer_path = os.path.join(save_dir, "tokenizer.pkl")
        labels_path = os.path.join(save_dir, "labels.pkl")
        
        logger.info("Saving model locally...")
        model.save(model_path)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        with open(labels_path, "wb") as f:
            pickle.dump(label_encoder.classes_, f)
        
        logger.info(f"Model saved locally to: {save_dir}")

        # Save to Google Cloud Storage if available
        if gcs_manager:
            logger.info("Attempting to save model to Google Cloud Storage...")
            logger.info(f"GCS Manager enabled: {gcs_manager.enabled}")
            logger.info(f"GCS Bucket: {gcs_manager.bucket_name if hasattr(gcs_manager, 'bucket_name') else 'Not set'}")
            
            gcs_success = gcs_manager.save_model(client_id, model, tokenizer, label_encoder.classes_)
            if gcs_success:
                logger.info("✓ Model successfully saved to Google Cloud Storage")
                
                # Verify the files were actually uploaded
                try:
                    prefix = "global" if not client_id else f"client_{client_id}"
                    blob_model = gcs_manager.bucket.blob(f"{prefix}/model.keras")
                    blob_tokenizer = gcs_manager.bucket.blob(f"{prefix}/tokenizer.pkl")
                    blob_labels = gcs_manager.bucket.blob(f"{prefix}/labels.pkl")
                    
                    logger.info(f"GCS Verification - Model exists: {blob_model.exists()}")
                    logger.info(f"GCS Verification - Tokenizer exists: {blob_tokenizer.exists()}")
                    logger.info(f"GCS Verification - Labels exist: {blob_labels.exists()}")
                    
                except Exception as verify_error:
                    logger.error(f"GCS verification failed: {verify_error}")
                    
            else:
                logger.error("✗ Failed to save model to Google Cloud Storage")
        else:
            logger.warning("Google Cloud Storage manager not available")

        logger.info("Training completed successfully")
        logger.info(f"Vocabulary size: {len(tokenizer.word_index)}")
        logger.info(f"Number of classes: {len(label_encoder.classes_)}")
        
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=str, default=None)
    args = parser.parse_args()

    logger.info(f"Starting training for client: {args.client_id or 'global'}")
    
    # Log environment variables for debugging (mask sensitive values)
    logger.info(f"GCS_BUCKET_NAME: {os.environ.get('GCS_BUCKET_NAME')}")
    logger.info(f"GOOGLE_APPLICATION_CREDENTIALS_JSON present: {bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))}")
    
    if not train_model(client_id=args.client_id):
        logger.error("Training failed")
        sys.exit(1)
    
    logger.info("Training completed successfully")
    sys.exit(0)