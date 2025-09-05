
import os
import json
import pickle
import random
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mysql.connector
import mysql.connector.pooling
import subprocess
import threading
import uuid
import logging
import time
from datetime import datetime
import sys
from typing import Dict, Tuple, Optional, List
import base64
import re
import gzip
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorFlow for optimized performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')
os.makedirs('data/models', exist_ok=True)

# Optimize TensorFlow memory usage
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.set_soft_device_placement(True)

app = Flask(__name__)
CORS(app)

# Try to import Google Cloud Storage
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    from google.api_core.exceptions import GoogleAPIError
    GCS_AVAILABLE = True
    logger.info("Google Cloud Storage libraries available")
except ImportError:
    logger.warning("Google Cloud Storage libraries not available")
    GCS_AVAILABLE = False

class Config:
    """Application configuration"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join('data', 'models')
    MIN_CONFIDENCE = 0.05
    TRAINING_TIMEOUT = 3600  # 60 minutes

    # Database configuration with connection pooling
    DB_HOST = os.environ.get('DB_HOST', '104.243.44.92')
    DB_NAME = os.environ.get('DB_NAME', 'eddcode1_aidexchatbot')
    DB_USER = os.environ.get('DB_USER', 'eddcode1_talkbot')
    DB_PASS = os.environ.get('DB_PASS', 'ZS.nxgC^&9%Bc4E8')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    DB_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', 5))
    DB_POOL_RECYCLE = int(os.environ.get('DB_POOL_RECYCLE', 3600))
    
    # GCS configuration
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'client-chatbot-api-models')
    
    # Cache configuration
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 300))  # 5 minutes
    MAX_CACHE_SIZE = int(os.environ.get('MAX_CACHE_SIZE', 1000))  # Max cached responses

    # Performance settings
    PREDICTION_TIMEOUT = int(os.environ.get('PREDICTION_TIMEOUT', 3))  # 3 seconds timeout
    MODEL_PRELOAD_WORKERS = int(os.environ.get('MODEL_PRELOAD_WORKERS', 2))

    @classmethod
    def get_model_path(cls, client_id=None):
        """Returns absolute path to model directory"""
        base_path = os.path.abspath(os.path.join(cls.BASE_DIR, cls.MODELS_DIR))
        return os.path.join(base_path, "global" if not client_id else str(client_id))
    
    @classmethod
    def get_model_files(cls, client_id=None):
        """Returns dict of expected model file paths"""
        model_dir = cls.get_model_path(client_id)
        return {
            "model": os.path.join(model_dir, "model.keras"),
            "tokenizer": os.path.join(model_dir, "tokenizer.pkl"),
            "labels": os.path.join(model_dir, "labels.pkl"),
        }


# Global training status
training_in_progress = False
training_lock = threading.Lock()

# Database connection pool
db_pool = None

# Response cache
response_cache = {}
cache_lock = threading.Lock()

def init_db_pool():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="chatbot_pool",
            pool_size=Config.DB_POOL_SIZE,
            pool_reset_session=True,
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASS,
            port=Config.DB_PORT,
            connection_timeout=5,
            connect_timeout=5,
            auth_plugin='mysql_native_password',
            autocommit=True
        )
        logger.info(f"Database connection pool initialized with {Config.DB_POOL_SIZE} connections")
        return True
    except mysql.connector.Error as err:
        logger.error(f"MySQL Error: {err}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        return False

def get_db_connection():
    """Get database connection from pool with retry logic"""
    global db_pool
    
    # If pool doesn't exist, try to initialize it
    if db_pool is None:
        if not init_db_pool():
            return None
    
    try:
        connection = db_pool.get_connection()
        if connection.is_connected():
            return connection
        else:
            logger.error("Got connection from pool but it's not connected")
            return None
    except mysql.connector.Error as err:
        logger.error(f"MySQL Error getting connection: {err}")
        # Try to reinitialize pool on error
        try:
            init_db_pool()
            return db_pool.get_connection() if db_pool else None
        except:
            return None
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        return None

def test_db_connection():
    """Test database connection directly (bypass pool)"""
    try:
        connection = mysql.connector.connect(
            host=Config.DB_HOST,
            database=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASS,
            port=Config.DB_PORT,
            connection_timeout=5,
            auth_plugin='mysql_native_password'
        )
        if connection.is_connected():
            connection.close()
            return True
        return False
    except mysql.connector.Error as err:
        logger.error(f"MySQL Connection Test Error: {err}")
        return False
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

class GoogleCloudStorageManager:
    def __init__(self):
        self.enabled = False
        try:
            if not GCS_AVAILABLE:
                logger.warning("GCS libraries not available")
                return
                
            if not Config.GCS_BUCKET_NAME:
                logger.warning("GCS_BUCKET_NAME not set, Google Cloud Storage disabled")
                return
            
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
                    return
            else:
                # Use default credentials (when running on Google Cloud)
                self.client = storage.Client()
            
            # Get bucket
            self.bucket = self.client.bucket(Config.GCS_BUCKET_NAME)
            if not self.bucket.exists():
                logger.warning(f"Bucket {Config.GCS_BUCKET_NAME} does not exist")
                return
            
            self.enabled = True
            logger.info(f"Google Cloud Storage manager initialized with bucket: {Config.GCS_BUCKET_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage: {e}")
            self.enabled = False
    
    def download_file(self, remote_path, local_path):
        """Download a file from Google Cloud Storage"""
        if not self.enabled:
            return False
            
        try:
            blob = self.bucket.blob(remote_path)
            if not blob.exists():
                logger.warning(f"File not found in GCS: {remote_path}")
                return False
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{Config.GCS_BUCKET_NAME}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            return False
    
    def download_model(self, client_id):
        """Download model components from GCS"""
        if not self.enabled:
            return False
            
        try:
            prefix = "global" if not client_id else f"client_{client_id}"
            model_dir = Config.get_model_path(client_id)
            os.makedirs(model_dir, exist_ok=True)
            
            success = True
            success &= self.download_file(f"{prefix}/model.keras", os.path.join(model_dir, "model.keras"))
            success &= self.download_file(f"{prefix}/tokenizer.pkl", os.path.join(model_dir, "tokenizer.pkl"))
            success &= self.download_file(f"{prefix}/labels.pkl", os.path.join(model_dir, "labels.pkl"))
            
            return success
                
        except Exception as e:
            logger.error(f"Failed to download model from GCS: {e}")
            return False

# Initialize GCS manager
gcs_manager = GoogleCloudStorageManager() if GCS_AVAILABLE else None

class ModelCache:
    """In-memory cache for loaded models with TTL"""
    _instance = None
    _cache: Dict[Tuple[str, str], Tuple[tf.keras.Model, object, list, float]] = {}
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls, client_id: str, model_type: str):
        """Get cached model if not expired"""
        key = (client_id or "global", model_type)
        with cls._lock:
            if key in cls._cache:
                model, tokenizer, labels, timestamp = cls._cache[key]
                # Check if cache entry is still valid
                if time.time() - timestamp < Config.CACHE_TIMEOUT:
                    return model, tokenizer, labels
                else:
                    # Remove expired entry
                    del cls._cache[key]
            return None

    @classmethod
    def set(cls, client_id: str, model_type: str, model: tf.keras.Model, tokenizer: object, labels: list):
        """Cache a model with current timestamp"""
        key = (client_id or "global", model_type)
        with cls._lock:
            cls._cache[key] = (model, tokenizer, labels, time.time())

    @classmethod
    def clear(cls):
        """Clear cache"""
        with cls._lock:
            cls._cache.clear()

def load_model_from_disk(client_id=None):
    """Load model from disk files with optimized loading"""
    try:
        model_files = Config.get_model_files(client_id)
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in model_files.values()):
            logger.warning(f"Model files not found for client {client_id}")
            return None
        
        # Load from disk with optimized settings
        with tf.device('/cpu:0'):
            # Load model with optimized settings
            model = tf.keras.models.load_model(
                model_files["model"], 
                compile=False
            )
            
            # Load tokenizer and labels
            with open(model_files["tokenizer"], "rb") as f:
                tokenizer = pickle.load(f)
            with open(model_files["labels"], "rb") as f:
                labels = pickle.load(f)
        
        logger.info(f"Successfully loaded model for client {client_id}")
        return model, tokenizer, labels
    except Exception as e:
        logger.error(f"Error loading model from disk: {e}")
        logger.error(traceback.format_exc())
        return None

def preload_models():
    """Preload models on startup to reduce first request latency"""
    logger.info("Preloading models...")
    
    # Get all client IDs from database
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT user_token FROM chatbot_intents WHERE user_token IS NOT NULL")
            client_ids = [row[0] for row in cursor.fetchall()]
            client_ids.append(None)  # Add global model
            
            # Preload each model
            for client_id in client_ids[:3]:  # Limit to first 3 to avoid overload
                try:
                    ensure_model_loaded(client_id)
                    logger.info(f"Preloaded model for client: {client_id or 'global'}")
                except Exception as e:
                    logger.error(f"Failed to preload model for {client_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error getting client IDs for preloading: {e}")
        finally:
            if conn and conn.is_connected():
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {str(e)}")
    else:
        logger.warning("Cannot preload models - database connection failed")
    
    logger.info("Model preloading completed")

def ensure_model_loaded(client_id):
    """Ensure model is loaded, try cache first, then disk, then GCS"""
    # Check if model is in cache
    cached = ModelCache.get(client_id or "global", "main")
    if cached:
        return cached
    
    # Try to load from disk
    model_data = load_model_from_disk(client_id)
    if model_data:
        ModelCache.set(client_id or "global", "main", *model_data)
        return model_data
    
    # Try to download from Google Cloud Storage
    if gcs_manager and gcs_manager.enabled:
        try:
            if gcs_manager.download_model(client_id):
                model_data = load_model_from_disk(client_id)
                if model_data:
                    ModelCache.set(client_id or "global", "main", *model_data)
                    return model_data
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
    
    # If client-specific model fails, try global model
    if client_id:
        return ensure_model_loaded(None)
    
    return None

def clean_text(text):
    """Clean and preprocess text for better prediction"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Basic text normalization
    text = re.sub(r'[^\w\s\?\.\!]', '', text)
    
    return text


def predict_intent(msg: str, client_id: str = None) -> Tuple[str, float]:
    """Predict intent from message with optimized processing"""
    try:
        logger.info(f"Loading model for client: {client_id}")
        
        # Get model data
        model_data = ensure_model_loaded(client_id)
        if not model_data:
            logger.error(f"No model data found for client {client_id}")
            return "fallback", 0.0
        
        model, tokenizer, labels = model_data
        
        # Preprocess message
        clean_msg = clean_text(msg.lower().strip())
        logger.info(f"Processing message: '{clean_msg}'")
        
        # Process prediction with batch optimization
        sequence = tokenizer.texts_to_sequences([clean_msg])
        if not sequence or not sequence[0]:
            logger.warning(f"No sequence generated for message: '{clean_msg}'")
            return "fallback", 0.0
        
        # Get the actual sequence length from the model
        try:
            if hasattr(model, 'input_shape') and model.input_shape:
                max_len = model.input_shape[1]
            else:
                # Fallback to a reasonable default
                max_len = 20
                logger.warning(f"Using default max_len: {max_len}")
        except:
            max_len = 20
            logger.warning(f"Error getting max_len, using default: {max_len}")
            
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Make prediction
        prediction = model.predict(padded, verbose=0, batch_size=1)
        
        confidence = float(np.max(prediction))
        intent_idx = np.argmax(prediction)
        
        # Check if intent_idx is within bounds
        if intent_idx < len(labels):
            intent = labels[intent_idx]
        else:
            logger.error(f"Intent index {intent_idx} out of bounds for labels length {len(labels)}")
            intent = "fallback"
            
        logger.info(f"Predicted intent: {intent} with confidence: {confidence}")
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return "error", 0.0

def _run_training(client_id, training_id):
    global training_in_progress

    model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
    os.makedirs(model_dir, exist_ok=True)

    # define first, outside try
    required_files = [
        os.path.join(model_dir, "model.keras"),
        os.path.join(model_dir, "tokenizer.pkl"),
        os.path.join(model_dir, "labels.pkl")
    ]

    training_successful = False

    try:
        ModelCache.clear()
        cmd = [sys.executable, "train.py"]
        if client_id:
            cmd.extend(["--client-id", client_id])

        logger.info(f"Starting training in {model_dir}")
        log_file = os.path.join(model_dir, f"training_{training_id}.log")

        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=open(log_file, 'w'),
            stderr=subprocess.STDOUT,
            text=True
        )

        start_time = time.time()
        process.wait(timeout=Config.TRAINING_TIMEOUT)

        if process.returncode != 0:
            with open(log_file) as f:
                log_content = f.read()
            raise Exception(f"Training failed with return code {process.returncode}. Logs: {log_content[-1000:]}")

        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise Exception(f"Missing output files: {missing}")

        training_time = time.time() - start_time
        logger.info(f"Training completed successfully in {training_time:.2f} seconds")
        training_successful = True

    except subprocess.TimeoutExpired:
        process.kill()
        logger.error("Training timed out")
        raise

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        if not training_successful:
            for f in required_files:
                if os.path.exists(f):
                    os.remove(f)
                    logger.info(f"Cleaned up partial file: {f}")
        raise
    finally:
        training_in_progress = False

def get_cached_response(intent, client_id):
    """Get response from cache if available"""
    cache_key = f"{client_id or 'global'}:{intent}"
    with cache_lock:
        cached_data = response_cache.get(cache_key)
        if cached_data and time.time() - cached_data['timestamp'] < Config.CACHE_TIMEOUT:
            return cached_data['response']
    return None

def cache_response(intent, client_id, response):
    """Cache a response"""
    cache_key = f"{client_id or 'global'}:{intent}"
    with cache_lock:
        # Implement simple LRU cache eviction if needed
        if len(response_cache) >= Config.MAX_CACHE_SIZE:
            # Remove oldest entry
            oldest_key = min(response_cache.keys(), key=lambda k: response_cache[k]['timestamp'])
            del response_cache[oldest_key]
            
        response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }

def get_response(intent, client_id=None):
    """Get response for intent with improved error handling and caching"""
    # Check cache first
    cached_response = get_cached_response(intent, client_id)
    if cached_response:
        return {
            "response": cached_response,
            "status": "success",
            "source": "cache"
        }
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to establish database connection")
            return {
                "response": "I'm having connection issues. Please try again later.",
                "status": "db_error",
                "error": "Database connection failed"
            }

        # Build queries with proper parameterization
        queries = []
        params = []
        
        # 1. First try client-specific response if client_id provided
        if client_id:
            queries.append("""
                SELECT responses FROM chatbot_intents 
                WHERE tag = %s AND user_token = %s
                ORDER BY created_at DESC
                LIMIT 1
            """)
            params.append((intent, client_id))
        
        # 2. Always include global responses as fallback
        queries.append("""
            SELECT responses FROM chatbot_intents 
            WHERE tag = %s AND user_token IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """)
        params.append((intent,))

        response = None
        source = None
        with conn.cursor(dictionary=True) as cursor:
            for i, (query, param) in enumerate(zip(queries, params)):
                try:
                    cursor.execute(query, param)
                    if result := cursor.fetchone():
                        responses = json.loads(result["responses"])
                        response = random.choice(responses)
                        source = "client" if i == 0 else "global"
                        logger.info(f"Found response for intent '{intent}' from {source}")
                        break
                except Exception as e:
                    logger.warning(f"Query failed: {query} with params {param}. Error: {str(e)}")
                    continue

        if response:
            # Cache the response
            cache_response(intent, client_id, response)
            return {
                "response": response,
                "status": "success",
                "source": source
            }
        
        logger.warning(f"No response found for intent: {intent}")
        return {
            "response": "I'm not sure how to respond to that. Could you rephrase?",
            "status": "no_response",
            "intent": intent
        }

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in responses: {str(e)}")
        return {
            "response": "There seems to be a problem with my response data.",
            "status": "data_error",
            "error": "Invalid response format"
        }
    except Exception as e:
        logger.error(f"Unexpected error in get_response: {str(e)}", exc_info=True)
        return {
            "response": "Sorry, I encountered an error processing your request.",
            "status": "error",
            "error": str(e)
        }
    finally:
        if conn and conn.is_connected():
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")

def compress_response(f):
    """Decorator to compress responses with gzip"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        
        # Check if client supports gzip
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
        
        # Compress response data
        if response.status_code == 200 and response.content_length > 500:  # Only compress larger responses
            compressed_data = gzip.compress(response.get_data())
            response.set_data(compressed_data)
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Length'] = len(compressed_data)
        
        return response
    return decorated_function

@app.before_request
def before_request():
    """Measure request timing"""
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Log request timing"""
    if hasattr(request, 'start_time'):
        duration = time.time() - request.start_time
        logger.info(f"Request {request.method} {request.path} took {duration:.3f}s")
        response.headers['X-Response-Time'] = f'{duration:.3f}s'
    return response

@app.errorhandler(500)
def handle_500_error(e):
    """Handle 500 errors with detailed logging"""
    logger.error(f"500 Error: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal Server Error",
        "message": "The server encountered an internal error",
        "status": 500
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status": 500
    }), 500

@app.route("/predict", methods=["POST", "OPTIONS"])
@compress_response
def predict():
    """Prediction endpoint with optimized response"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
    # Check Content-Type header
    if request.content_type != 'application/json':
        return jsonify({
            "error": "Unsupported Media Type",
            "message": "Content-Type must be application/json"
        }), 415
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "Bad Request",
                "message": "No JSON data received"
            }), 400
            
        msg = data.get("msg", "").strip()
        client_id = data.get("clientId")
        
        if not msg:
            return jsonify({
                "error": "Bad Request", 
                "message": "Message cannot be empty"
            }), 400
            
        logger.info(f"Received prediction request: '{msg}' for client: {client_id}")
            
        # Start prediction - with detailed error handling
        try:
            intent, confidence = predict_intent(msg, client_id)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a fallback response instead of crashing
            return jsonify({
                "response": "I'm having trouble processing your request right now. Please try again.",
                "intent": "fallback",
                "confidence": 0.0,
                "status": "service_error"
            })
        
        if confidence < Config.MIN_CONFIDENCE:
            return jsonify({
                "response": "I'm not quite sure what you mean. Could you rephrase?",
                "intent": intent,
                "confidence": round(confidence, 2),
                "status": "low_confidence"
            })

        # Get response from database
        try:
            response = get_response(intent, client_id)
            return jsonify({
                "response": response["response"],
                "intent": intent,
                "confidence": round(confidence, 2),
                "status": response["status"]
            })
        except Exception as e:
            logger.error(f"Response lookup failed: {str(e)}")
            # Fallback response if database is unavailable
            return jsonify({
                "response": "I understand what you're asking, but I'm having trouble retrieving the response.",
                "intent": intent,
                "confidence": round(confidence, 2),
                "status": "response_error"
            })

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to process request",
            "status": 500
        }), 500
    
@app.route("/train", methods=["POST", "OPTIONS"])
def start_training():
    """Initiate training process with detailed responses"""
    global training_in_progress
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid request data",
                "details": "Expected JSON payload"
            }), 400
            
        client_id = data.get("clientId")
        
        with training_lock:
            if training_in_progress:
                return jsonify({
                    "status": "error",
                    "message": "Training already in progress",
                    "details": "Only one training session can run at a time"
                }), 429

            training_id = str(uuid.uuid4())
            training_in_progress = True
            
            # Start training in background thread
            threading.Thread(
                target=_run_training,
                args=(client_id, training_id),
                daemon=True
            ).start()
            
            return jsonify({
                "status": "started",
                "message": "Training initiated successfully",
                "training_id": training_id,
                "details": {
                    "client_id": client_id,
                    "start_time": datetime.now().isoformat(),
                    "estimated_time": "Typically takes 3-5 minutes"
                }
            })

    except Exception as e:
        logger.error(f"Training initialization failed: {str(e)}", exc_info=True)
        training_in_progress = False
        return jsonify({
            "status": "error",
            "message": "Failed to start training",
            "details": str(e)
        }), 500

@app.route("/training-complete", methods=["GET"])
def check_training_complete():
    """Check if training is complete"""
    with training_lock:
        return jsonify({
            "status": "success" if not training_in_progress else "running",
            "message": "Training completed successfully" if not training_in_progress else "Training in progress",
            "details": {
                "last_training_time": datetime.now().isoformat() if not training_in_progress else None
            }
        })

@app.route("/check-models", methods=["GET"])
def check_models():
    """Enhanced model verification endpoint that checks both local and GCS"""
    try:
        client_id = request.args.get("clientId")
        prefix = "global" if not client_id else f"client_{client_id}"
        
        logger.info(f"Checking models for client: {client_id or 'global'}")
        
        # Check local filesystem first
        model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
        abs_path = os.path.abspath(model_dir)
        
        logger.info(f"Checking local model path: {abs_path}")

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        required_files = {
            "model": os.path.join(model_dir, "model.keras"),
            "tokenizer": os.path.join(model_dir, "tokenizer.pkl"), 
            "labels": os.path.join(model_dir, "labels.pkl")
        }

        # Check for local files
        local_existing = {}
        local_missing = []
        for name, path in required_files.items():
            if os.path.exists(path):
                local_existing[name] = {
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path),
                    "location": "local"
                }
            else:
                local_missing.append(name)

        # Check Google Cloud Storage if configured
        gcs_existing = {}
        gcs_missing = []
        gcs_available = False
        
        if gcs_manager and gcs_manager.enabled:
            gcs_available = True
            try:
                # Check if files exist in GCS
                gcs_files = [
                    f"{prefix}/model.keras",
                    f"{prefix}/tokenizer.pkl",
                    f"{prefix}/labels.pkl"
                ]
                
                for i, gcs_file in enumerate(gcs_files):
                    file_type = list(required_files.keys())[i]
                    try:
                        blob = gcs_manager.bucket.blob(gcs_file)
                        if blob.exists():
                            # Get file metadata
                            blob.reload()
                            gcs_existing[file_type] = {
                                "size": blob.size,
                                "modified": blob.updated.timestamp() if blob.updated else None,
                                "location": "gcs",
                                "gcs_path": gcs_file
                            }
                        else:
                            gcs_missing.append(file_type)
                    except Exception as e:
                        logger.warning(f"Error checking GCS file {gcs_file}: {e}")
                        gcs_missing.append(file_type)
                        
            except Exception as e:
                logger.error(f"Error checking GCS: {e}")
                gcs_available = False

        # Determine overall status
        all_local_files_exist = len(local_missing) == 0
        all_gcs_files_exist = len(gcs_missing) == 0
        any_files_exist = len(local_existing) > 0 or len(gcs_existing) > 0

        if all_local_files_exist:
            # All files exist locally - verify integrity
            try:
                tf.keras.models.load_model(required_files["model"])
                with open(required_files["tokenizer"], "rb") as f:
                    pickle.load(f)
                with open(required_files["labels"], "rb") as f:
                    pickle.load(f)
                    
                return jsonify({
                    "status": "success",
                    "model_status": "ready",
                    "location": "local",
                    "files": local_existing,
                    "gcs_available": gcs_available,
                    "gcs_files": gcs_existing if gcs_available else {}
                })
                
            except Exception as e:
                logger.error(f"Local model verification failed: {str(e)}")
                return jsonify({
                    "status": "error",
                    "model_status": "corrupted",
                    "location": "local",
                    "error": str(e),
                    "suggested_action": "Retrain model or download from GCS"
                }), 500

        elif all_gcs_files_exist and gcs_available:
            # All files exist in GCS but not locally
            return jsonify({
                "status": "success",
                "model_status": "available_in_gcs",
                "location": "gcs",
                "files": gcs_existing,
                "local_files": local_existing,
                "message": "Models available in Google Cloud Storage but not locally",
                "suggested_action": "Download models from GCS using /download-models endpoint"
            })

        elif not any_files_exist:
            # No files exist anywhere
            return jsonify({
                "status": "error",
                "model_status": "not_found",
                "local_files": local_existing,
                "gcs_files": gcs_existing if gcs_available else {},
                "gcs_available": gcs_available,
                "message": "No model files found locally or in GCS",
                "suggested_action": "Run training first"
            }), 404

        else:
            # Some files exist but not all
            return jsonify({
                "status": "error",
                "model_status": "incomplete",
                "local_missing_files": local_missing,
                "local_existing_files": local_existing,
                "gcs_missing_files": gcs_missing if gcs_available else [],
                "gcs_existing_files": gcs_existing if gcs_available else {},
                "gcs_available": gcs_available,
                "message": "Some model files are missing",
                "suggested_action": "Retrain model or download missing files from GCS"
            }), 404

    except Exception as e:
        logger.error(f"Model check failed: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e),
            "suggested_action": "Check server logs"
        }), 500

@app.route("/download-models", methods=["POST"])
def download_models_from_gcs():
    """Download models from Google Cloud Storage"""
    if not gcs_manager or not gcs_manager.enabled:
        return jsonify({
            "status": "error",
            "message": "Google Cloud Storage not enabled"
        }), 400
    
    try:
        client_id = request.json.get("clientId") if request.is_json else None
        success = gcs_manager.download_model(client_id)
        
        if success:
            # Clear cache to force reload of new models
            ModelCache.clear()
            return jsonify({
                "status": "success",
                "message": "Models downloaded from Google Cloud Storage"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to download models from GCS"
            }), 404
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/check-data", methods=["GET"])
def check_data():
    """Check if training data exists"""
    conn = None
    try:
        client_id = request.args.get("clientId")
        conn = get_db_connection()
        if not conn:
            return jsonify({
                "status": "error",
                "message": "Database connection failed",
                "data_status": "unavailable"
            }), 500
            
        query = "SELECT COUNT(*) as count FROM chatbot_intents WHERE user_token = %s OR user_token IS NULL"
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            
        if result["count"] > 0:
            return jsonify({
                "status": "success",
                "message": "Data available",
                "data_status": "available",
                "count": result["count"]
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No training data found",
                "data_status": "unavailable"
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "data_status": "error"
        }), 500
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.route("/gcs-status", methods=["GET"])
def check_gcs_status():
    """Check Google Cloud Storage status"""
    try:
        if not gcs_manager:
            return jsonify({
                "status": "error",
                "message": "Google Cloud Storage manager not initialized",
                "gcs_configured": False,
                "gcs_enabled": False
            }), 500
        
        if not gcs_manager.enabled:
            return jsonify({
                "status": "error",
                "message": "Google Cloud Storage not enabled",
                "gcs_configured": bool(Config.GCS_BUCKET_NAME),
                "gcs_enabled": False
            }), 500
        
        # Test connectivity
        try:
            # Try to list a few files to test connectivity
            blobs = list(gcs_manager.client.list_blobs(gcs_manager.bucket, max_results=1))
            connected = True
            error_msg = None
        except Exception as e:
            connected = False
            error_msg = str(e)
        
        return jsonify({
            "status": "success" if connected else "error",
            "gcs_configured": True,
            "gcs_enabled": True,
            "gcs_connected": connected,
            "bucket_name": Config.GCS_BUCKET_NAME,
            "error": error_msg if not connected else None
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"GCS status check failed: {str(e)}",
            "gcs_configured": bool(Config.GCS_BUCKET_NAME),
            "gcs_enabled": False
        }), 500

@app.route("/training-logs", methods=["GET"])
def get_training_logs():
    """Get the last training logs"""
    try:
        # Check the most recent training process
        log_files = []
        training_dir = os.path.join('data', 'models')
        
        if os.path.exists(training_dir):
            for root, dirs, files in os.walk(training_dir):
                for file in files:
                    if file.endswith('.log') or file.endswith('.txt'):
                        log_files.append(os.path.join(root, file))
        
        logs = []
        for log_file in log_files[-5:]:  # Get last 5 log files
            try:
                with open(log_file, 'r') as f:
                    logs.append({
                        'file': log_file,
                        'content': f.read()[-2000:]  # Last 2000 characters
                    })
            except:
                pass
        
        return jsonify({
            "status": "success",
            "log_files": log_files,
            "recent_logs": logs
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check - try database connection
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        conn.close()
        
        # Check GCS status
        gcs_enabled = gcs_manager and gcs_manager.enabled if gcs_manager else False
        gcs_configured = bool(Config.GCS_BUCKET_NAME)
        
        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "tensorflow_version": tf.__version__,
                "training_in_progress": training_in_progress,
                "gcs": {
                    "configured": gcs_configured,
                    "enabled": gcs_enabled
                }
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route("/cache-stats", methods=["GET"])
def cache_stats():
    """Get cache statistics"""
    with cache_lock:
        cache_size = len(response_cache)
        cache_keys = list(response_cache.keys())[:10]  # First 10 keys
        
    with ModelCache._lock:
        model_cache_size = len(ModelCache._cache)
        model_cache_keys = list(ModelCache._cache.keys())[:5]  # First 5 keys
    
    return jsonify({
        "response_cache": {
            "size": cache_size,
            "sample_keys": cache_keys
        },
        "model_cache": {
            "size": model_cache_size,
            "sample_keys": model_cache_keys
        }
    })

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Clear all caches"""
    with cache_lock:
        response_cache.clear()
    
    ModelCache.clear()
    
    return jsonify({
        "status": "success",
        "message": "All caches cleared"
    })

# Initialize the application
def initialize_app():
    """Initialize the application asynchronously"""
    logger.info("Starting application initialization...")
    
    # Initialize database pool
    init_db_pool()
    
    # Start model preloading in background
    def preload_models_async():
        try:
            preload_models()
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
    
    # Start preloading in background thread
    preload_thread = threading.Thread(target=preload_models_async, daemon=True)
    preload_thread.start()
    
    logger.info("Application initialization completed")

# Initialize the app
initialize_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    
    # For Google Cloud Run, use this simple server
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)