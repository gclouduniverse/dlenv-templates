locals {
  vm_names = [
    for i in range(tonumber(var.gce_vm_count)):
      "${var.gce_vm_name}-${i}"
  ]
}

resource "google_compute_instance" "vms" {

  for_each = toset(local.vm_names)

  name = each.key
  machine_type = var.gce_machine_type

  guest_accelerator {
    type = var.gce_gpu_type
    count = var.gce_gpu_count
  }

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/${var.dlvm_image}"
    }
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  network_interface {
    network = "default"
    access_config {
    }
  }

  metadata = {
    install-nvidia-driver = "True",
    // if var.gce_ssh_pub_key_file is specified, pass the ssh public key to the instance metadata.
    sshKeys = var.gce_ssh_pub_key_file != "" ? "${var.gce_ssh_user}:${file(var.gce_ssh_pub_key_file)}" : null
  }
}
