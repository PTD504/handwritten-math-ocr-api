PROJECT_ID="your-project-id"
SERVICE_NAME="latex-ocr-api"
REGION="us-central1"

# Create alerting policy cho errors
gcloud alpha monitoring policies create --policy-from-file=- <<EOF
{
  "displayName": "High Error Rate - ${SERVICE_NAME}",
  "conditions": [
    {
      "displayName": "Error rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/request_count\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_SUM",
            "groupByFields": ["resource.labels.service_name"]
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 10,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "enabled": true
}
EOF

# Create alerting policy cho memory usage
gcloud alpha monitoring policies create --policy-from-file=- <<EOF
{
  "displayName": "High Memory Usage - ${SERVICE_NAME}",
  "conditions": [
    {
      "displayName": "Memory usage condition",
      "conditionThreshold": {
        "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"${SERVICE_NAME}\" AND metric.type=\"run.googleapis.com/container/memory/utilizations\"",
        "aggregations": [
          {
            "alignmentPeriod": "300s",
            "perSeriesAligner": "ALIGN_MEAN",
            "crossSeriesReducer": "REDUCE_MEAN",
            "groupByFields": ["resource.labels.service_name"]
          }
        ],
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.8,
        "duration": "300s"
      }
    }
  ],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "enabled": true
}
EOF

echo "Monitoring alerts đã được tạo"
echo "Xem tại: https://console.cloud.google.com/monitoring/alerting"