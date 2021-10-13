A simple example to run a TensorFlow 2.6 DL container on a Kubernetes cluster.
It creates a pod and a service.

You can use port-forwarding to access the jupyter notebook running on the pod.
```
kubectl apply -f dlcontainer.yaml
kubectl port-forward jupyter-pod 8888:8080
```
