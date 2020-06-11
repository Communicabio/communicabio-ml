PROJECT_ID=stunning-hull-187717

docker build . -t gcr.io/${PROJECT_ID}/gpt2_server

docker login -u oauth2accesstoken -p "$(gcloud auth print-access-token)" https://gcr.io

docker push gcr.io/${PROJECT_ID}/gpt2_server

gcloud container clusters create gpt2cluster \
  --accelerator=type=nvidia-tesla-k80 \
  --machine-type=n1-standard-2 \
  --disk-size=20Gb \
  --preemptible \
  --enable-autoscaling --max-nodes=3 --min-nodes=0 \
  --num-nodes=1 \
  --region=us-west1-b

kubectl create deployment gpt2-server --image=gcr.io/${PROJECT_ID}/gpt2_server:v10
kubectl create deployment gpt2-server --image=gcr.io/google-samples/hello-app:1.0

kubectl expose deployment gpt2-server --type=LoadBalancer --port 80 --target-port 8080

kubectl get events
kubectl get pods

kubectl delete service gpt2-server

gcloud container clusters delete gpt2cluster --region=us-west1-b
