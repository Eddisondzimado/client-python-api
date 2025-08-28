import os
import pickle
import tempfile
import logging
from google.cloud import storage
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

class GoogleCloudStorageManager:
    def __init__(self):
        try:
            # Get bucket name from environment or use default
            self.bucket_name = os.environ.get('GCS_BUCKET_NAME', 'your-project-id-chatbot-models')
            
            # Check if we have credentials
            creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if creds_json:
                # Use provided credentials
                credentials_info = json.loads(creds_json)
                self.credentials = service_account.Credentials.from_service_account_info(credentials_info)
                self.client = storage.Client(credentials=self.credentials)
            else:
                # Use default credentials (when running on Google Cloud)
                self.client = storage.Client()
            
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test connection
            if not self.bucket.exists():
                logger.warning(f"Bucket {self.bucket_name} does not exist. Attempting to create...")
                self.bucket = self.client.create_bucket(self.bucket_name, location='us-central1')
            
            logger.info(f"Google Cloud Storage manager initialized with bucket: {self.bucket_name}")
            self.enabled = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Storage: {e}")
            self.enabled = False
            self.client = None
            self.bucket = None
    
    def upload_file(self, local_path, remote_path):
        """Upload a file to Google Cloud Storage"""
        if not self.enabled:
            logger.warning("Google Cloud Storage not enabled")
            return False
            
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return False
    
    def download_file(self, remote_path, local_path):
        """Download a file from Google Cloud Storage"""
        if not self.enabled:
            logger.warning("Google Cloud Storage not enabled")
            return False
            
        try:
            blob = self.bucket.blob(remote_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            return False
    
    def file_exists(self, remote_path):
        """Check if a file exists in Google Cloud Storage"""
        if not self.enabled:
            return False
            
        try:
            blob = self.bucket.blob(remote_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking file existence in GCS: {e}")
            return False
    
    def save_model(self, client_id, model, tokenizer, label_encoder):
        """Save model components to GCS"""
        if not self.enabled:
            return False
            
        try:
            prefix = "global" if not client_id else f"client_{client_id}"
            
            # Save to temporary files first
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model.keras")
                tokenizer_path = os.path.join(temp_dir, "tokenizer.pkl")
                labels_path = os.path.join(temp_dir, "labels.pkl")
                
                # Save files locally
                model.save(model_path)
                with open(tokenizer_path, "wb") as f:
                    pickle.dump(tokenizer, f)
                with open(labels_path, "wb") as f:
                    pickle.dump(label_encoder, f)
                
                # Upload to GCS
                success = True
                success &= self.upload_file(model_path, f"{prefix}/model.keras")
                success &= self.upload_file(tokenizer_path, f"{prefix}/tokenizer.pkl")
                success &= self.upload_file(labels_path, f"{prefix}/labels.pkl")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to save model to GCS: {e}")
            return False
    
    def load_model(self, client_id):
        """Load model components from GCS"""
        if not self.enabled:
            return None
            
        try:
            prefix = "global" if not client_id else f"client_{client_id}"
            
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = os.path.join(temp_dir, "model.keras")
                tokenizer_path = os.path.join(temp_dir, "tokenizer.pkl")
                labels_path = os.path.join(temp_dir, "labels.pkl")
                
                # Download files from GCS
                success = True
                success &= self.download_file(f"{prefix}/model.keras", model_path)
                success &= self.download_file(f"{prefix}/tokenizer.pkl", tokenizer_path)
                success &= self.download_file(f"{prefix}/labels.pkl", labels_path)
                
                if not success:
                    return None
                
                # Load files
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
                with open(tokenizer_path, "rb") as f:
                    tokenizer = pickle.load(f)
                
                with open(labels_path, "rb") as f:
                    labels = pickle.load(f)
                
                return model, tokenizer, labels
                
        except Exception as e:
            logger.error(f"Failed to load model from GCS: {e}")
            return None

# Global instance
gcs_manager = GoogleCloudStorageManager()