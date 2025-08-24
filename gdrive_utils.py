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
            # Get credentials from environment variable
            creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
            if not creds_json:
                raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not set")
            
            # Get folder ID from environment variable
            self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
            if not self.folder_id:
                raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable not set")
            
            logger.info(f"Using Google Drive folder ID: {self.folder_id}")
            
            # Parse the JSON from environment variable
            self.credentials_info = json.loads(creds_json)
            
            # Create credentials
            self.credentials = service_account.Credentials.from_service_account_info(
                self.credentials_info,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            # Build the service
            self.service = build('drive', 'v3', credentials=self.credentials)
            
            # Test connection
            self.test_connection()
            
            logger.info("Google Drive manager initialized successfully")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in GOOGLE_DRIVE_CREDENTIALS: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive manager: {e}")
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
            logger.info(f"Google Drive connection test successful. Found {len(files)} files in folder.")
            return True
            
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            raise
    
    def upload_file(self, local_path, remote_name, mime_type='application/octet-stream'):
        """Upload a file to Google Drive"""
        try:
            # Check if file exists locally
            if not os.path.exists(local_path):
                logger.error(f"Local file not found: {local_path}")
                return None
            
            file_metadata = {
                'name': remote_name,
                'parents': [self.folder_id]
            }
            
            media = MediaFileUpload(local_path, mimetype=mime_type)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"Uploaded {remote_name} to Google Drive with ID: {file_id}")
            return file_id
            
        except HttpError as e:
            logger.error(f"Google Drive API error uploading {remote_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to upload {remote_name} to Google Drive: {e}")
            return None
    
    def download_file(self, remote_name, local_path):
        """Download a file from Google Drive"""
        try:
            # Search for the file
            query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name)"
            ).execute()
            files = results.get('files', [])
            
            if not files:
                logger.error(f"File {remote_name} not found in Google Drive")
                return False
            
            file_id = files[0]['id']
            
            # Download the file
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {int(status.progress() * 100)}%")
            
            # Save to local file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                f.write(fh.getvalue())
            
            logger.info(f"Downloaded {remote_name} from Google Drive to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {remote_name} from Google Drive: {e}")
            return False
    
    def file_exists(self, remote_name):
        """Check if a file exists in Google Drive"""
        try:
            query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                fields="files(id)"
            ).execute()
            return len(results.get('files', [])) > 0
        except Exception as e:
            logger.error(f"Failed to check if file exists in Google Drive: {e}")
            return False
    
    def delete_file(self, remote_name):
        """Delete a file from Google Drive"""
        try:
            query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query, 
                fields="files(id)"
            ).execute()
            files = results.get('files', [])
            
            if files:
                self.service.files().delete(fileId=files[0]['id']).execute()
                logger.info(f"Deleted {remote_name} from Google Drive")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file from Google Drive: {e}")
            return False

# Global instance with proper error handling
try:
    gdrive_manager = GoogleDriveManager()
    logger.info("Google Drive manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Drive manager: {e}")
    gdrive_manager = None