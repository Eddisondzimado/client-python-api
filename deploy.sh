@echo off
setlocal enabledelayedexpansion

set PROJECT_ID=client-support-chatbot-api
set SERVICE_NAME=customer-support-api
set REGION=us-central1
set BUCKET_NAME=client-support-chatbot-api.appspot.com

echo Enabling required APIs...
gcloud services enable ^
    cloudbuild.googleapis.com ^
    run.googleapis.com ^
    containerregistry.googleapis.com ^
    --project=%PROJECT_ID%

echo Setting up permissions...
for /f "tokens=*" %%i in ('gcloud projects describe %PROJECT_ID% --format="value(projectNumber)"') do set PROJECT_NUMBER=%%i

gcloud projects add-iam-policy-binding %PROJECT_ID% ^
    --member="serviceAccount:!PROJECT_NUMBER!@cloudbuild.gserviceaccount.com" ^
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding %PROJECT_ID% ^
    --member="serviceAccount:!PROJECT_NUMBER!@cloudbuild.gserviceaccount.com" ^
    --role="roles/iam.serviceAccountUser"

echo Submitting build to Google Cloud Build...
gcloud builds submit ^
    --config=cloudbuild.yaml ^
    --substitutions="_BUCKET_NAME=%BUCKET_NAME%,_SERVICE_NAME=%SERVICE_NAME%,_REGION=%REGION%" ^
    --project=%PROJECT_ID%

if !errorlevel! neq 0 (
    echo Build failed!
    exit /b 1
)

echo Deployment complete!
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region %REGION% --format="value(status.url)" 2^>nul') do set SERVICE_URL=%%i
if defined SERVICE_URL (
    echo Service URL: !SERVICE_URL!
) else (
    echo Could not retrieve service URL
)

endlocal
