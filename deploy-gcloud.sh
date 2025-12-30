#!/bin/bash
# Deploy PCB Thermal AI to Google Cloud Run

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="pcb-thermal-ai"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying PCB Thermal AI to Google Cloud Run"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Service: ${SERVICE_NAME}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth print-identity-token &> /dev/null; then
    echo "📝 Please authenticate with Google Cloud:"
    gcloud auth login
fi

# Set project
echo "📁 Setting project to ${PROJECT_ID}..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com containerregistry.googleapis.com run.googleapis.com

# Build and push Docker image
echo "🐳 Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "PYTHONUNBUFFERED=1"

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your app is live at: ${SERVICE_URL}"
echo ""
echo "To set Gemini API key (optional, for AI PCB generation):"
echo "  gcloud run services update ${SERVICE_NAME} --region ${REGION} --set-env-vars GEMINI_API_KEY=your-key"
