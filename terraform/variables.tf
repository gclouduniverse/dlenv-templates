variable "gce_vm_name" {
  description = "Name prefix of the VMs."
}
variable "gce_project_id" {
  description = "Name of the project to use for instantiating clusters."
}
variable "gce_ssh_pub_key_file" {
  description = "SSH public key file path."
  default = ""
}
variable "gce_ssh_user" {
  description = "SSH username."
  default = ""
}
variable "dlvm_image" {
  description = "Deep Learning VM family name."
  default = "tf-2-6-cu110"
}
variable "gce_vm_count" {
  description = "Number of VMs."
  default = "2"
}
variable "gce_region" {
  description = "Region to instantiating clusters."
  default = "us-west1"
}
variable "gce_machine_type" {
  description = "VM machine type."
  default = "n1-highmem-8"
}
variable "gce_gpu_type" {
  description = "GPU type."
  default = "nvidia-tesla-t4"
}
variable "gce_gpu_count" {
  description = "Number of GPUs."
  default = "1"
}
