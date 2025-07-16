# Set up script for deploying a Handwritten Math OCR API on Google Cloud Run
PROJECT_ID="your-project-id"
REGION="asia-southeast1"
SERVICE_NAME="service-name"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN} Bắt đầu triển khai ${SERVICE_NAME}${NC}"

# Check if gcloud is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED} Chưa đăng nhập gcloud. Chạy: gcloud auth login${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW} Thiết lập project: ${PROJECT_ID}${NC}"
gcloud config set project ${PROJECT_ID}

# Enable necessary APIs
echo -e "${YELLOW} Kích hoạt APIs cần thiết...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable redis.googleapis.com

# Check Redis instance
echo -e "${YELLOW} Kiểm tra Redis instance...${NC}"
REDIS_HOST=$(gcloud redis instances describe latex-ocr-redis --region=${REGION} --format="value(host)" 2>/dev/null)
if [ -z "$REDIS_HOST" ]; then
    echo -e "${YELLOW}  Tạo Redis instance...${NC}"
    gcloud redis instances create latex-ocr-redis \
        --size=1 \
        --region=${REGION} \
        --redis-version=redis_6_x \
        --network=default
    
    # Wait for Redis instance to be ready
    echo -e "${YELLOW} Đợi Redis instance sẵn sàng...${NC}"
    while [ -z "$REDIS_HOST" ]; do
        sleep 10
        REDIS_HOST=$(gcloud redis instances describe latex-ocr-redis --region=${REGION} --format="value(host)" 2>/dev/null)
    done
fi

echo -e "${GREEN} Redis host: ${REDIS_HOST}${NC}"

# Create API key in Secret Manager
if ! gcloud secrets describe model-api-key >/dev/null 2>&1; then
    echo -e "${YELLOW} Tạo API key...${NC}"
    API_KEY=$(openssl rand -hex 32)
    echo -n "$API_KEY" | gcloud secrets create model-api-key --data-file=-
    echo -e "${GREEN} API key đã được tạo và lưu vào Secret Manager${NC}"
else
    echo -e "${GREEN} API key đã tồn tại${NC}"
fi

# Docker setup
echo -e "${YELLOW} Cấu hình Docker...${NC}"
gcloud auth configure-docker

# Build image
echo -e "${YELLOW} Build Docker image...${NC}"
docker build -t ${IMAGE_NAME}:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED} Build image thất bại${NC}"
    exit 1
fi

# Push image
echo -e "${YELLOW} Push image lên Container Registry...${NC}"
docker push ${IMAGE_NAME}:latest

if [ $? -ne 0 ]; then
    echo -e "${RED} Push image thất bại${NC}"
    exit 1
fi

# Deploy on Cloud Run
echo -e "${YELLOW} Deploy lên Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},ENVIRONMENT=production,REDIS_URL=redis://${REDIS_HOST}:6379,RATE_LIMIT_PER_MINUTE=20,RATE_LIMIT_PER_HOUR=200,RATE_LIMIT_PER_DAY=1000,CONCURRENT_REQUESTS=5" \
    --set-secrets="MODEL_API_KEY=model-api-key:latest"

if [ $? -ne 0 ]; then
    echo -e "${RED} Deploy thất bại${NC}"
    exit 1
fi

# Get URL service
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format="value(status.url)")

echo -e "${GREEN} Triển khai thành công!${NC}"
echo -e "${GREEN} Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN} Health check: ${SERVICE_URL}/health${NC}"
echo -e "${GREEN} Status: ${SERVICE_URL}/status${NC}"

# Test health check
echo -e "${YELLOW} Kiểm tra health check...${NC}"
sleep 5
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${SERVICE_URL}/health")
if [ "$HEALTH_STATUS" -eq 200 ]; then
    echo -e "${GREEN} Health check thành công${NC}"
else
    echo -e "${RED} Health check thất bại (HTTP ${HEALTH_STATUS})${NC}"
fi

echo -e "${GREEN} Hoàn thành triển khai!${NC}"
echo -e "${YELLOW} Lưu ý: Thay đổi PROJECT_ID trong script trước khi chạy${NC}"