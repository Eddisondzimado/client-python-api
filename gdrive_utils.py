
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError

class GoogleDriveManager:
    def __init__(self):
        # Get credentials from environment variable ONLY
        creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
        if not creds_json:
            raise ValueError("GOOGLE_DRIVE_CREDENTIALS environment variable not set")
        
        # Parse the JSON from environment variable
        self.credentials_info = json.loads(creds_json)
        
        # Get folder ID from environment variable
        self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
        if not self.folder_id:
            raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable not set")
        
        # Create credentials
        self.credentials = service_account.Credentials.from_service_account_info(
            self.credentials_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        # Build the service
        self.service = build('drive', 'v3', credentials=self.credentials)
        
        # Create credentials
        self.credentials = service_account.Credentials.from_service_account_info(
            self.credentials_info,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        
        # Build the service
        self.service = build('drive', 'v3', credentials=self.credentials)
    
    def upload_file(self, local_path, remote_name, mime_type='application/octet-stream'):
        """Upload a file to Google Drive"""
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
        
        return file.get('id')
    
    def download_file(self, remote_name, local_path):
        """Download a file from Google Drive"""
        # Search for the file
        query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if not files:
            raise FileNotFoundError(f"File {remote_name} not found in Google Drive")
        
        file_id = files[0]['id']
        
        # Download the file
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Save to local file
        with open(local_path, 'wb') as f:
            f.write(fh.getvalue())
        
        return True
    
    def file_exists(self, remote_name):
        """Check if a file exists in Google Drive"""
        query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id)").execute()
        return len(results.get('files', [])) > 0
    
    def delete_file(self, remote_name):
        """Delete a file from Google Drive"""
        query = f"name='{remote_name}' and '{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id)").execute()
        files = results.get('files', [])
        
        if files:
            self.service.files().delete(fileId=files[0]['id']).execute()
            return True
        return False

# Global instance
try:
    gdrive_manager = GoogleDriveManager()
except Exception as e:
    print(f"Failed to initialize Google Drive manager: {e}")
    gdrive_manager = None