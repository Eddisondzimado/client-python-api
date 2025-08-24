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
import subprocess
import threading
import uuid
from werkzeug.utils import secure_filename
import logging
import time
from datetime import datetime
import sys
import functools
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorFlow for optimized performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Optimize TensorFlow memory usage
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

app = Flask(__name__)
CORS(app)

# Google Drive Configuration
try:
    from gdrive_utils import gdrive_manager
    GOOGLE_DRIVE_ENABLED = True
    logger.info("Google Drive integration enabled")
except ImportError:
    GOOGLE_DRIVE_ENABLED = False
    logger.warning("Google Drive integration disabled - gdrive_utils not found")

class Config:
    """Application configuration"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join('data', 'models')  # Changed to relative path
    MIN_CONFIDENCE = 0.05
    TRAINING_TIMEOUT = 600  # 10 minutes

    # Database configuration
    DB_HOST = os.environ.get('DB_HOST', '104.243.44.92')
    DB_NAME = os.environ.get('DB_NAME', 'eddcode1_aidexchatbot')
    DB_USER = os.environ.get('DB_USER', 'eddcode1_talkbot')
    DB_PASS = os.environ.get('DB_PASS', '(ruaN^{r)7I&')
    DB_PORT = int(os.environ.get('DB_PORT', 3306))
    

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


def get_db_connection():
    """Get database connection with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = mysql.connector.connect(
                host=Config.DB_HOST,
                database=Config.DB_NAME,
                user=Config.DB_USER,
                password=Config.DB_PASS,
                port=Config.DB_PORT,
                connect_timeout=5
            )
            return conn
        except Exception as e:
            logger.error(f"DB connection attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)

class ModelCache:
    """In-memory cache for loaded models"""
    _instance = None
    _cache: Dict[Tuple[str, str], Tuple[tf.keras.Model, object, list]] = {}  # (client_id, model_type) -> (model, tokenizer, labels)
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get(cls, client_id: str, model_type: str):
        """Get cached model"""
        key = (client_id or "global", model_type)
        with cls._lock:
            return cls._cache.get(key)

    @classmethod
    def set(cls, client_id: str, model_type: str, model: tf.keras.Model, tokenizer: object, labels: list):
        """Cache a model"""
        key = (client_id or "global", model_type)
        with cls._lock:
            cls._cache[key] = (model, tokenizer, labels)

    @classmethod
    def clear(cls):
        """Clear cache"""
        with cls._lock:
            cls._cache.clear()

def download_from_google_drive(client_id):
    """Download model files from Google Drive if they don't exist locally"""
    if not GOOGLE_DRIVE_ENABLED:
        logger.warning("Google Drive not enabled, skipping download")
        return False
    
    try:
        prefix = "global" if not client_id else f"client_{client_id}"
        model_files = Config.get_model_files(client_id)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_files["model"]), exist_ok=True)
        
        # Download files from Google Drive
        gdrive_manager.download_file(f"{prefix}_model.keras", model_files["model"])
        gdrive_manager.download_file(f"{prefix}_tokenizer.pkl", model_files["tokenizer"])
        gdrive_manager.download_file(f"{prefix}_labels.pkl", model_files["labels"])
        
        logger.info(f"Downloaded model files for {prefix} from Google Drive")
        return True
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
        return False

# Modified predict_intent function with caching
@functools.lru_cache(maxsize=32)  # Cache recent predictions
def predict_intent(msg: str, client_id: str = None) -> Tuple[str, float]:
    """Predict intent from message using cached models"""
    try:
        # Get from cache or load fresh
        cache_key = client_id or "global"
        cached = ModelCache.get(cache_key, "main")
        
        if not cached:
            # If not in cache, try to load from disk or Google Drive
            model_files = Config.get_model_files(client_id)
            
            # Check if we need to download from Google Drive
            need_download = False
            for file_type in ["model", "tokenizer", "labels"]:
                local_path = model_files[file_type]
                if not os.path.exists(local_path):
                    need_download = True
                    break
            
            if need_download:
                download_success = download_from_google_drive(client_id)
                if not download_success:
                    if client_id:  # Fallback to global model
                        return predict_intent(msg)
                    return "fallback", 0.0
            
            # Now load from local disk
            with tf.device('/cpu:0'):
                model = tf.keras.models.load_model(model_files["model"], compile=False)
                with open(model_files["tokenizer"], "rb") as f:
                    tokenizer = pickle.load(f)
                with open(model_files["labels"], "rb") as f:
                    labels = pickle.load(f)
                
                # Store in cache for future use
                ModelCache.set(cache_key, "main", model, tokenizer, labels)
                cached = (model, tokenizer, labels)
        
        model, tokenizer, labels = cached
        
        # Process the prediction
        sequence = tokenizer.texts_to_sequences([msg.lower().strip()])
        padded = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
        prediction = model.predict(padded, verbose=0)
        
        confidence = float(np.max(prediction))
        intent = labels[np.argmax(prediction)]
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "error", 0.0
    
def _run_training(client_id, training_id):
    """Enhanced training process with automatic cache loading"""
    global training_in_progress
    
    try:
        model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
        os.makedirs(model_dir, exist_ok=True)
        
        # Clear any cached models for this client before training
        ModelCache.clear()
        
        cmd = [sys.executable, "train.py"]
        if client_id:
            cmd.extend(["--client-id", client_id])
            
        logger.info(f"Starting training in {model_dir}")
        
        process = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=Config.TRAINING_TIMEOUT)
            
            if process.returncode != 0:
                raise Exception(stderr or stdout or "Training failed")
                
            # Verify output files
            required_files = [
                os.path.join(model_dir, "model.keras"),
                os.path.join(model_dir, "tokenizer.pkl"),
                os.path.join(model_dir, "labels.pkl")
            ]
            
            missing = [f for f in required_files if not os.path.exists(f)]
            if missing:
                raise Exception(f"Missing output files: {missing}")
                
            logger.info("Training completed successfully")
            
            # NEW: Load the trained models into cache immediately
            try:
                with tf.device('/cpu:0'):
                    model = tf.keras.models.load_model(required_files[0], compile=False)
                    with open(required_files[1], "rb") as f:
                        tokenizer = pickle.load(f)
                    with open(required_files[2], "rb") as f:
                        labels = pickle.load(f)
                    
                    ModelCache.set(client_id or "global", "main", model, tokenizer, labels)
                    logger.info(f"Successfully cached model for client: {client_id or 'global'}")
                    
            except Exception as cache_error:
                logger.error(f"Failed to cache model: {str(cache_error)}")
                # This isn't fatal - the system will load from disk later
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Training timed out")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Clean up partial files
        for f in required_files:
            if os.path.exists(f):
                os.remove(f)
        raise
    finally:
        training_in_progress = False

# Add cache management endpoint
@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear model cache"""
    try:
        ModelCache.clear()
        return jsonify({
            "status": "success",
            "message": "Model cache cleared",
            "cache_size": 0
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/cache/status", methods=["GET"])
def cache_status():
    """Get cache status"""
    try:
        return jsonify({
            "status": "success",
            "cache_size": len(ModelCache._cache),
            "cached_models": list(ModelCache._cache.keys())
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def get_response(intent, client_id=None):
    """Get response for intent with improved error handling"""
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
        with conn.cursor(dictionary=True) as cursor:
            for query, param in zip(queries, params):
                try:
                    cursor.execute(query, param)
                    if result := cursor.fetchone():
                        responses = json.loads(result["responses"])
                        response = random.choice(responses)
                        logger.info(f"Found response for intent '{intent}'")
                        break
                except Exception as e:
                    logger.warning(f"Query failed: {query} with params {param}. Error: {str(e)}")
                    continue

        if response:
            return {
                "response": response,
                "status": "success",
                "source": "client" if client_id and query == queries[0] else "global"
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

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Prediction endpoint"""
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
            
        intent, confidence = predict_intent(msg, client_id)
        
        if confidence < Config.MIN_CONFIDENCE:
            return jsonify({
                "response": "I'm not quite sure what you mean. Could you rephrase?",
                "intent": intent,
                "confidence": round(confidence, 2),
                "status": "low_confidence"
            })

        response = get_response(intent, client_id)
        return jsonify({
            "response": response,
            "intent": intent,
            "confidence": round(confidence, 2),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to process request"
        }), 500
    

@app.route("/train", methods=["POST", "OPTIONS"])
def start_training():
    """Initiate training process with detailed responses"""
    global training_in_progress  # Add this line
    
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
        training_in_progress = False  # Ensure we reset on error
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
    """Enhanced model verification endpoint"""
    try:
        client_id = request.args.get("clientId")
        model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
        abs_path = os.path.abspath(model_dir)
        
        logger.info(f"Checking model path: {abs_path}")

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        required_files = {
            "model": os.path.join(model_dir, "model.keras"),
            "tokenizer": os.path.join(model_dir, "tokenizer.pkl"), 
            "labels": os.path.join(model_dir, "labels.pkl")
        }

        # Check for files
        existing = {}
        missing = []
        for name, path in required_files.items():
            if os.path.exists(path):
                existing[name] = {
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path)
                }
            else:
                missing.append(name)

        if missing:
            logger.warning(f"Missing model files: {missing}")
            # Check if this is first-time training
            is_first_time = not any(existing.values())
            return jsonify({
                "status": "error",
                "message": "First-time training required" if is_first_time else "Model files incomplete",
                "model_status": "first_time" if is_first_time else "incomplete",
                "missing_files": missing,
                "directory_contents": os.listdir(model_dir),
                "suggested_action": "Run training" if is_first_time else "Retrain or check model files"
            }), 404

        # Verify file integrity
        try:
            tf.keras.models.load_model(required_files["model"])
            with open(required_files["tokenizer"], "rb") as f:
                pickle.load(f)
            with open(required_files["labels"], "rb") as f:
                pickle.load(f)
                
            return jsonify({
                "status": "success",
                "model_status": "ready",
                "files": existing
            })
            
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return jsonify({
                "status": "error",
                "model_status": "corrupted",
                "error": str(e),
                "suggested_action": "Retrain model"
            }), 500

    except Exception as e:
        logger.error(f"Model check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "suggested_action": "Check server logs"
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

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check - try database connection
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        conn.close()
        
        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "tensorflow_version": tf.__version__,
                "training_in_progress": training_in_progress,
                "google_drive_enabled": GOOGLE_DRIVE_ENABLED
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)