PROJECT_ID=stunning-hull-187717
IMAGE=gpt2_server

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE



PROJECT_ID=stunning-hull-187717
VERSION=3

docker build . -t gcr.io/${PROJECT_ID}/gpt2_server_cpu:$VERSION

docker login -u oauth2accesstoken -p "$(gcloud auth print-access-token)" https://gcr.io

docker push gcr.io/${PROJECT_ID}/gpt2_server_cpu:$VERSION

gcloud container clusters create gpt2cluster-cpu \
  --machine-type=custom-4-6144 \
  --disk-size=20Gb \
  --preemptible \
  --enable-autoscaling --max-nodes=3 --min-nodes=0 \
  --region=us-central1-c \
  --enable-cloud-logging \
  --num-nodes=1 \
  --enable-cloud-monitoring


kubectl create deployment gpt2-service-cpu --image=gcr.io/${PROJECT_ID}/gpt2_server_cpu:$VERSION
kubectl expose deployment gpt2-service-cpu --type=LoadBalancer --port 80 --target-port 8080

kubectl get events
kubectl get pods

kubectl delete service gpt2-service-cpu

gcloud container clusters delete gpt2cluster-cpu --region=us-central1-c

kubectl exec -it gpt2-server-84b4468f4b-m6mjz -- /bin/bash
