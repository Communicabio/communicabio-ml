PROJECT_ID=stunning-hull-187717
IMAGE=en_gpt2_server

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE
