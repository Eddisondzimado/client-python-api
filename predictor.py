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
import logging
import time
from datetime import datetime
import gc
import threading
import subprocess
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorFlow for memory optimization
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
CORS(app)

# Global training status
training_in_progress = False
training_lock = threading.Lock()

# Google Drive Configuration
GOOGLE_DRIVE_ENABLED = False
gdrive_manager = None

# Try to import Google Drive dependencies
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    
    class GoogleDriveManager:
        def __init__(self):
            try:
                creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
                if not creds_json:
                    raise ValueError("GOOGLE_DRIVE_CREDENTIALS not set")
                
                self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
                if not self.folder_id:
                    raise ValueError("GOOGLE_DRIVE_FOLDER_ID not set")
                
                self.credentials_info = json.loads(creds_json)
                self.credentials = service_account.Credentials.from_service_account_info(
                    self.credentials_info,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                
                self.service = build('drive', 'v3', credentials=self.credentials)
                logger.info("Google Drive manager initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize Google Drive: {e}")
                raise
        
        def download_file(self, remote_name, local_path):
            """Download a file from Google Drive"""
            try:
                query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
                results = self.service.files().list(q=query, fields="files(id, name)").execute()
                files = results.get('files', [])
                
                if not files:
                    raise FileNotFoundError(f"File {remote_name} not found")
                
                file_id = files[0]['id']
                request = self.service.files().get_media(fileId=file_id)
                
                with open(local_path, 'wb') as f:
                    downloader = MediaIoBaseDownload(f, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                
                logger.info(f"Downloaded {remote_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to download {remote_name}: {e}")
                return False

        def file_exists(self, remote_name):
            """Check if a file exists on Google Drive"""
            try:
                query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
                results = self.service.files().list(q=query, fields="files(id, name)").execute()
                files = results.get('files', [])
                return len(files) > 0
            except Exception as e:
                logger.error(f"Error checking file existence: {e}")
                return False

    # Initialize Google Drive manager
    try:
        gdrive_manager = GoogleDriveManager()
        GOOGLE_DRIVE_ENABLED = True
        logger.info("Google Drive integration enabled")
    except Exception as e:
        logger.warning(f"Google Drive initialization failed: {e}")

except ImportError:
    logger.warning("Google Drive dependencies not available")
except Exception as e:
    logger.warning(f"Error loading Google Drive: {e}")

class Config:
    """Application configuration"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join('data', 'models')
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
        base_path = os.path.abspath(os.path.join(cls.BASE_DIR, cls.MODELS_DIR))
        return os.path.join(base_path, "global" if not client_id else str(client_id))
    
    @classmethod
    def get_model_files(cls, client_id=None):
        model_dir = cls.get_model_path(client_id)
        return {
            "model": os.path.join(model_dir, "model.keras"),
            "tokenizer": os.path.join(model_dir, "tokenizer.pkl"),
            "labels": os.path.join(model_dir, "labels.pkl"),
        }

class ModelCache:
    """In-memory cache for loaded models"""
    _cache = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, client_id):
        key = client_id or "global"
        with cls._lock:
            return cls._cache.get(key)

    @classmethod
    def set(cls, client_id, model, tokenizer, labels):
        key = client_id or "global"
        with cls._lock:
            cls._cache[key] = (model, tokenizer, labels)

    @classmethod
    def clear(cls, client_id=None):
        with cls._lock:
            if client_id:
                key = client_id or "global"
                if key in cls._cache:
                    del cls._cache[key]
            else:
                cls._cache.clear()

def download_from_google_drive(client_id):
    """Download model files from Google Drive"""
    if not GOOGLE_DRIVE_ENABLED:
        return False
    
    try:
        prefix = "global" if not client_id else f"client_{client_id}"
        model_files = Config.get_model_files(client_id)
        
        os.makedirs(os.path.dirname(model_files["model"]), exist_ok=True)
        
        success = True
        for file_type, remote_name in [
            ("model", f"{prefix}_model.keras"),
            ("tokenizer", f"{prefix}_tokenizer.pkl"),
            ("labels", f"{prefix}_labels.pkl")
        ]:
            if not gdrive_manager.download_file(remote_name, model_files[file_type]):
                success = False
        
        return success
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def load_model_from_disk(client_id):
    """Load model from disk files"""
    try:
        model_files = Config.get_model_files(client_id)
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in model_files.values()):
            return None
        
        # Load from disk
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(model_files["model"], compile=False)
            with open(model_files["tokenizer"], "rb") as f:
                tokenizer = pickle.load(f)
            with open(model_files["labels"], "rb") as f:
                labels = pickle.load(f)
        
        return model, tokenizer, labels
    except Exception as e:
        logger.error(f"Error loading model from disk: {e}")
        return None

def ensure_model_loaded(client_id):
    """Ensure model is loaded, download from Google Drive if needed"""
    # Check if model is in cache
    cached = ModelCache.get(client_id)
    if cached:
        return cached
    
    # Try to load from disk
    model_data = load_model_from_disk(client_id)
    if model_data:
        ModelCache.set(client_id, *model_data)
        return model_data
    
    # Download from Google Drive if available
    try:
        if gdrive_manager and download_from_google_drive(client_id):
            model_data = load_model_from_disk(client_id)
            if model_data:
                ModelCache.set(client_id, *model_data)
                return model_data
    except Exception as e:
        logger.error(f"Failed to download from Google Drive: {e}")
    
    # If client-specific model fails, try global model
    if client_id:
        return ensure_model_loaded(None)
    
    return None

def predict_intent(msg, client_id=None):
    """Predict intent from message"""
    try:
        # Get model data
        model_data = ensure_model_loaded(client_id)
        if not model_data:
            return "fallback", 0.0
        
        model, tokenizer, labels = model_data
        
        # Process prediction
        sequence = tokenizer.texts_to_sequences([msg.lower().strip()])
        padded = pad_sequences(sequence, maxlen=model.input_shape[1], padding='post')
        prediction = model.predict(padded, verbose=0)
        
        confidence = float(np.max(prediction))
        intent = labels[np.argmax(prediction)]
        return intent, confidence
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "error", 0.0

def get_db_connection():
    """Get database connection with retry logic"""
    for attempt in range(3):
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
            if attempt == 2:
                return None
            time.sleep(1)

def get_response(intent, client_id=None):
    """Get response for intent"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return {
                "response": "I'm having connection issues. Please try again later.",
                "status": "db_error"
            }

        queries, params = [], []
        
        if client_id:
            queries.append("SELECT responses FROM chatbot_intents WHERE tag = %s AND user_token = %s ORDER BY created_at DESC LIMit 1")
            params.append((intent, client_id))
        
        queries.append("SELECT responses FROM chatbot_intents WHERE tag = %s AND user_token IS NULL ORDER BY created_at DESC LIMit 1")
        params.append((intent,))

        response = None
        with conn.cursor(dictionary=True) as cursor:
            for query, param in zip(queries, params):
                try:
                    cursor.execute(query, param)
                    if result := cursor.fetchone():
                        responses = json.loads(result["responses"])
                        response = random.choice(responses)
                        break
                except Exception:
                    continue

        if response:
            return {
                "response": response,
                "status": "success",
                "source": "client" if client_id and query == queries[0] else "global"
            }
        
        return {
            "response": "I'm not sure how to respond to that. Could you rephrase?",
            "status": "no_response",
            "intent": intent
        }

    except Exception as e:
        logger.error(f"Response error: {e}")
        return {
            "response": "Sorry, I encountered an error processing your request.",
            "status": "error"
        }
    finally:
        if conn:
            conn.close()

def _run_training(client_id, training_id):
    """Training process with memory optimization"""
    global training_in_progress
    
    try:
        model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
        os.makedirs(model_dir, exist_ok=True)
        
        ModelCache.clear()
        
        cmd = ["python", "train.py"]
        if client_id:
            cmd.extend(["--client-id", client_id])
        
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
            
            if any(not os.path.exists(f) for f in required_files):
                raise Exception("Missing output files")
                
            logger.info("Training completed successfully")
            
            # Load into cache - FIXED: Only pass 4 arguments
            try:
                with tf.device('/cpu:0'):
                    model = tf.keras.models.load_model(required_files[0], compile=False)
                    with open(required_files[1], "rb") as f:
                        tokenizer = pickle.load(f)
                    with open(required_files[2], "rb") as f:
                        labels = pickle.load(f)
                    
                    # FIXED: Only pass 4 arguments to set()
                    ModelCache.set(client_id, model, tokenizer, labels)
                    
            except Exception as e:
                logger.error(f"Failed to cache model: {e}")
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise Exception("Training timed out")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clean up partial files
        for f in required_files:
            if os.path.exists(f):
                os.remove(f)
        raise
    finally:
        training_in_progress = False
        gc.collect()

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Prediction endpoint"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    
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
            "response": response["response"],
            "intent": intent,
            "confidence": round(confidence, 2),
            "status": response["status"]
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to process request"
        }), 500

@app.route("/train", methods=["POST", "OPTIONS"])
def start_training():
    """Initiate training process"""
    global training_in_progress
    
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid request data"
            }), 400
            
        client_id = data.get("clientId")
        
        with training_lock:
            if training_in_progress:
                return jsonify({
                    "status": "error",
                    "message": "Training already in progress"
                }), 429

            training_id = str(uuid.uuid4())
            training_in_progress = True
            
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
                    "start_time": datetime.now().isoformat()
                }
            })

    except Exception as e:
        logger.error(f"Training initialization failed: {e}")
        training_in_progress = False
        return jsonify({
            "status": "error",
            "message": "Failed to start training"
        }), 500

@app.route("/training-status", methods=["GET"])
def check_training_status():
    """Check if training is complete"""
    with training_lock:
        return jsonify({
            "status": "success" if not training_in_progress else "running",
            "message": "Training completed successfully" if not training_in_progress else "Training in progress",
            "training_in_progress": training_in_progress
        })

@app.route("/training-complete", methods=["GET"])
def check_training_complete():
    """Check if training is complete - alias for /training-status"""
    with training_lock:
        return jsonify({
            "status": "success" if not training_in_progress else "running",
            "message": "Training completed successfully" if not training_in_progress else "Training in progress",
            "training_in_progress": training_in_progress
        })

@app.route("/check-models", methods=["GET"])
def check_models():
    """Model verification endpoint"""
    try:
        client_id = request.args.get("clientId")
        model_dir = os.path.join('data', 'models', client_id if client_id else 'global')
        os.makedirs(model_dir, exist_ok=True)

        required_files = Config.get_model_files(client_id)
        existing, missing = {}, []
        
        for name, path in required_files.items():
            if os.path.exists(path):
                existing[name] = {"size": os.path.getsize(path)}
            else:
                missing.append(name)

        if missing:
            return jsonify({
                "status": "error",
                "message": "Model files incomplete",
                "missing_files": missing
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
            return jsonify({
                "status": "error",
                "model_status": "corrupted",
                "error": str(e)
            }), 500

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
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
                "message": "Database connection failed"
            }), 500
            
        query = "SELECT COUNT(*) as count FROM chatbot_intents WHERE user_token = %s OR user_token IS NULL"
        with conn.cursor(dictionary=True) as cursor:
            cursor.execute(query, (client_id,))
            result = cursor.fetchone()
            
        if result["count"] > 0:
            return jsonify({
                "status": "success",
                "message": "Data available",
                "count": result["count"]
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No training data found"
            }), 404
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()

@app.route("/gdrive-status", methods=["GET"])
def check_gdrive_status():
    """Check Google Drive connection status"""
    try:
        client_id = request.args.get("clientId")
        prefix = "global" if not client_id else f"client_{client_id}"
        
        if not GOOGLE_DRIVE_ENABLED:
            return jsonify({
                "status": "error",
                "message": "Google Drive integration not enabled. Please check environment variables.",
                "google_drive_connected": False,
                "files_on_gdrive": {},
                "configured": False
            }), 500
        
        # Check connection by listing files
        try:
            gdrive_manager.service.files().list(pageSize=1).execute()
            connection_ok = True
        except Exception as e:
            connection_ok = False
            logger.error(f"Google Drive connection test failed: {e}")
        
        # Check if model files exist on Google Drive
        files_exist = {}
        for file_type, remote_name in [
            ("model", f"{prefix}_model.keras"),
            ("tokenizer", f"{prefix}_tokenizer.pkl"),
            ("labels", f"{prefix}_labels.pkl")
        ]:
            try:
                files_exist[file_type] = gdrive_manager.file_exists(remote_name)
            except Exception as e:
                files_exist[file_type] = False
                logger.error(f"Error checking {remote_name}: {e}")
        
        return jsonify({
            "status": "success",
            "google_drive_connected": connection_ok,
            "files_on_gdrive": files_exist,
            "client_id": client_id,
            "configured": True
        })
        
    except Exception as e:
        logger.error(f"Google Drive status check failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Google Drive status check failed: {str(e)}",
            "google_drive_connected": False,
            "files_on_gdrive": {},
            "configured": False
        }), 500

@app.route("/upload-to-gdrive", methods=["POST", "OPTIONS"])
def upload_to_gdrive():
    """Manually upload model files to Google Drive"""
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400
            
        client_id = data.get("clientId")
        
        if not GOOGLE_DRIVE_ENABLED:
            return jsonify({
                "status": "error",
                "message": "Google Drive integration not enabled"
            }), 500
        
        # Get model files
        model_files = Config.get_model_files(client_id)
        
        # Check if files exist locally
        missing_files = []
        for file_type, path in model_files.items():
            if not os.path.exists(path):
                missing_files.append(file_type)
        
        if missing_files:
            return jsonify({
                "status": "error",
                "message": "Local model files missing",
                "missing_files": missing_files
            }), 404
        
        # Upload to Google Drive
        prefix = "global" if not client_id else f"client_{client_id}"
        
        upload_results = {}
        files_to_upload = [
            ("model", model_files["model"], f"{prefix}_model.keras"),
            ("tokenizer", model_files["tokenizer"], f"{prefix}_tokenizer.pkl"),
            ("labels", model_files["labels"], f"{prefix}_labels.pkl")
        ]
        
        for file_type, local_path, remote_name in files_to_upload:
            try:
                # Check if file already exists on Google Drive
                if gdrive_manager.file_exists(remote_name):
                    upload_results[file_type] = {
                        "status": "exists",
                        "message": f"File {remote_name} already exists on Google Drive"
                    }
                else:
                    file_id = gdrive_manager.upload_file(local_path, remote_name)
                    if file_id:
                        upload_results[file_type] = {
                            "status": "success",
                            "file_id": file_id,
                            "message": f"Uploaded {remote_name} successfully"
                        }
                    else:
                        upload_results[file_type] = {
                            "status": "error",
                            "message": f"Failed to upload {remote_name}"
                        }
            except Exception as e:
                upload_results[file_type] = {
                    "status": "error",
                    "message": f"Error uploading {remote_name}: {str(e)}"
                }
        
        # Check if all uploads were successful
        all_success = all(result["status"] in ["success", "exists"] for result in upload_results.values())
        
        return jsonify({
            "status": "success" if all_success else "partial",
            "message": "Google Drive upload completed",
            "results": upload_results,
            "client_id": client_id
        })
        
    except Exception as e:
        logger.error(f"Google Drive upload failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Google Drive upload failed: {str(e)}"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        conn.close()
        
        # Check Google Drive status with better error handling
        gdrive_status = {
            "enabled": GOOGLE_DRIVE_ENABLED,
            "connected": False,
            "error": None
        }
        
        if GOOGLE_DRIVE_ENABLED:
            try:
                gdrive_manager.service.files().list(pageSize=1).execute()
                gdrive_status["connected"] = True
            except Exception as e:
                gdrive_status["connected"] = False
                gdrive_status["error"] = str(e)
        else:
            gdrive_status["error"] = "Google Drive not configured. Check GOOGLE_DRIVE_CREDENTIALS and GOOGLE_DRIVE_FOLDER_ID environment variables."
        
        return jsonify({
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "tensorflow_version": tf.__version__,
                "google_drive": gdrive_status,
                "training_in_progress": training_in_progress
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