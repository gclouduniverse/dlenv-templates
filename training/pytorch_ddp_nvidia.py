"""PyTorch Distributed Data Parallel example from NVIDIA."""
# https://github.com/NVIDIA/DeepLearningExamples
import argparse

import utils
import virtual_machine


def main():
  parser = argparse.ArgumentParser(description='Optional app description')
  parser.add_argument('--vm-name', dest='vm_name', type=str, required=True)
  parser.add_argument('--vm-number', dest='vm_num', type=int, default=2)
  parser.add_argument('--gpu-per-vm', dest='gpu_per_vm', type=int, default=1)

  args = parser.parse_args()
  gpu_per_vm = args.gpu_per_vm
  vm_num = args.vm_num

  vms = []
  for i in range(args.vm_num):
    vm_name = '{}-{}'.format(args.vm_name, str(i))
    vms.append(virtual_machine.VirtualMachine(vm_name))

  # prepare
  prepare_command = """
  # install DALI
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

# install apex
pip install apex

# install nvidia-dllogger
pip install nvidia-pyindex
pip install nvidia-dllogger

# install pytorch-quantization
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/pytorch-quantization || exit
python setup.py install
cd || exit

# clone main repo
git clone https://github.com/NVIDIA/DeepLearningExamples.git
"""
  utils.run_threads(vms, [prepare_command] * len(vms), login_shell=True)

  # run
  master_ip = vms[0].internal_ip
  master_port = '1234'
  # pylint: disable=implicit-str-concat
  run_command_template = (
    'python '
    './DeepLearningExamples/PyTorch/Classification/ConvNets/multiproc.py '
    '--node_rank {node_rank} '
    '--master_addr {master_ip} '
    '--master_port {master_port} '
    '--nnodes {vm_num} '
    '--nproc_per_node {gpu_per_vm} '
    './DeepLearningExamples/PyTorch/Classification/ConvNets/main.py '
    '~/fake-data-path '
    '--data-backend syntetic '
    '--raport-file raport.json '
    '-j8 -p 100 '
    '--lr 4.096 '
    '--optimizer-batch-size 4096 '
    '--warmup 16 '
    '--arch resnet50 '
    '--label-smoothing 0.1 '
    '--lr-schedule cosine '
    '--mom 0.875 '
    '--wd 3.0517578125e-05 '
    '--no-checkpoints '
    '-b 256 '
    '--amp '
    '--static-loss-scale 128 '
    '--epochs 2'
  )

  commands = []
  for i, _ in enumerate(vms):
    run_command = run_command_template.format(
      node_rank=i,
      master_ip=master_ip,
      master_port=master_port,
      vm_num=vm_num,
      gpu_per_vm=gpu_per_vm,
    )
    commands.append(run_command)

  utils.run_threads(vms, commands, login_shell=True)


if __name__ == "__main__":
  main()
