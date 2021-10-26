terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  project = var.gce_project_id
  region = var.gce_region
  zone = "${var.gce_region}-b"
}
