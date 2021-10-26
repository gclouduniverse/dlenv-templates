## Distributed Training with DLVMs.

### Authentication

The easiest way to do this is to run `gcloud auth application-default login`.
 You can also use Google Cloud Service Account for authentication.
 For more details ib authentication, go to the [official Terraform guideline](https://registry.terraform.io/providers/hashicorp/google/latest/docs/guides/getting_started#configuring-the-provider).

For Horovod jobs, it requires the master VM to be able to SSH to all worker VMs. To create the needed SSH credentials, run
```shell script
ssh-keygen -t rsa -N '' -m PEM -q -f KEY_FILE_NAME
```
A public key and private key will be generated. The paths of these keys will be used in the following steps.

### Manage VMs with Terraform

The terraform configuration is define in `terraform/main.tf`. To initialize and preview,
```shell script
cd terraform
terraform init
terraform plan
```

To use PyTorch DLVM:
```shell script
terraform apply -var="gce_vm_name=test-vm" -var="gce_project_id=project1" -var="dlvm_image=pytorch-latest-gpu"
```

To use TensorFlow DLVMs (If running Horovod jobs, pass in the ssh credential as metadata):
```shell script
terraform apply -var="gce_vm_name=test-vm" -var="gce_project_id=project1" -var="dlvm_image=tf-2-6-cu110" -var="gce_ssh_pub_key_file=PUBLIC_KEY_PATH" -var="gce_ssh_user=USERNAME"
```

It will ask for confirmation. After typing `yes`, the VM cluster will be created. To verify the cluster, run `gcloud compute instances list`.

### Prepare training script.
Follow the examples in `example_trainers` to prepare training script. 

All scripts should be in one directory. The entry point should be named as `task.py`.

### Run distributed training jobs.

PyTorch Distributed Data Parallel (DDP) example:
```shell script
python pytorch_ddp.py --vm-name="test-vm" --src "example_trainers/pytorch_ddp"
```

The `--src "example_trainers/pytorch_ddp"` flag specify the path of the trainer script directory. By default, it uses the example trainer scripts.

TensorFlow Multi-worker strategy example:
```shell script
python tensorflow_multiworker.py --vm-name="test-vm"
```

Horovod TensorFlow example:
```shell script
python tensorflow_horovod.py --vm-name="test-vm" --private-key-path="PRIVATE_KEY_PATH" --user="USERNAME"
```
