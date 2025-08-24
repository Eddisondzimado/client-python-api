import os
import io
import json
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

# Set up logging
logger = logging.getLogger(__name__)

class GoogleDriveManager:
    def __init__(self):
        try:
            logger.info("Initializing Google Drive manager...")
            
            # Get credentials from environment variable
            creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
            logger.info(f"GOOGLE_DRIVE_CREDENTIALS found: {bool(creds_json)}")
            
            if not creds_json:
                raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not set or empty")
            
            # Get folder ID from environment variable
            self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
            logger.info(f"GOOGLE_DRIVE_FOLDER_ID found: {bool(self.folder_id)}")
            logger.info(f"Folder ID value: {self.folder_id}")
            
            if not self.folder_id:
                raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable not set")
            
            # Parse the JSON from environment variable
            try:
                self.credentials_info = json.loads(creds_json)
                logger.info("Successfully parsed credentials JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse credentials JSON: {e}")
                logger.error(f"JSON content (first 100 chars): {creds_json[:100]}...")
                raise
            
            # Check if required fields are present
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            for field in required_fields:
                if field not in self.credentials_info:
                    logger.error(f"Missing required field in credentials: {field}")
                    raise ValueError(f"Missing required field in credentials: {field}")
            
            logger.info("All required credential fields present")
            
            # Create credentials
            try:
                self.credentials = service_account.Credentials.from_service_account_info(
                    self.credentials_info,
                    scopes=['https://www.googleapis.com/auth/drive']
                )
                logger.info("Successfully created credentials")
            except Exception as e:
                logger.error(f"Failed to create credentials: {e}")
                raise
            
            # Build the service
            try:
                self.service = build('drive', 'v3', credentials=self.credentials)
                logger.info("Successfully built Google Drive service")
            except Exception as e:
                logger.error(f"Failed to build Google Drive service: {e}")
                raise
            
            # Test connection
            try:
                self.test_connection()
                logger.info("Google Drive connection test successful")
            except Exception as e:
                logger.error(f"Google Drive connection test failed: {e}")
                raise
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive manager: {e}", exc_info=True)
            raise
    
    def test_connection(self):
        """Test connection to Google Drive"""
        try:
            # Try to list files in the folder to test connectivity
            query = f"'{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in Google Drive folder")
            return True
            
        except HttpError as e:
            if e.resp.status == 404:
                logger.error("Google Drive folder not found or access denied. Please check:")
                logger.error("1. Folder ID is correct")
                logger.error("2. Folder is shared with the service account")
                logger.error(f"3. Service account email: {self.credentials_info.get('client_email')}")
            elif e.resp.status == 403:
                logger.error("Permission denied. Please check:")
                logger.error("1. Drive API is enabled for your project")
                logger.error("2. Service account has proper permissions")
            raise
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            raise

# Global instance with detailed error handling
try:
    gdrive_manager = GoogleDriveManager()
    logger.info("Google Drive manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Drive manager: {e}")
    gdrive_manager = None