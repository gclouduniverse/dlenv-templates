"""PyTorch DDP entry point."""

import argparse

import utils
import virtual_machine

MASTER_PORT = '1234'


def main():
  parser = argparse.ArgumentParser(description='Optional app description')
  parser.add_argument('--vm-name', dest='vm_name', type=str, required=True)
  parser.add_argument('--vm-number', dest='vm_num', type=int, default=2)
  parser.add_argument('--gpu-per-vm', dest='gpu_per_vm', type=int, default=1)
  parser.add_argument('--user', dest='user', type=str, default=None)
  parser.add_argument(
    '--src',
    dest='src',
    type=str,
    default='example_trainers/pytorch_ddp'
  )

  args = parser.parse_args()
  gpu_per_vm = args.gpu_per_vm
  vm_num = args.vm_num
  src = args.src

  vms = []
  for i in range(args.vm_num):
    vm_name = '{}-{}'.format(args.vm_name, str(i))
    vms.append(virtual_machine.VirtualMachine(vm_name))

  # prepare
  prepare_command = 'pip install apex'
  utils.run_threads(vms, [prepare_command] * len(vms), login_shell=True)

  for vm in vms:
    utils.scp(src, vm.name, 'trainer', user=args.user)

  # run
  master_ip = vms[0].internal_ip
  # pylint: disable=implicit-str-concat
  run_command_template = (
    'python '
    'trainer/task.py '
    '--nodes {vm_num} '
    '--nr {node_rank} '
    '--gpus {gpu_per_vm} '
    '--master_addr {master_ip} '
    '--master_port {master_port} '
    '--epochs 2'
  )

  commands = []
  for i, _ in enumerate(vms):
    run_command = run_command_template.format(
      vm_num=vm_num,
      node_rank=i,
      gpu_per_vm=gpu_per_vm,
      master_ip=master_ip,
      master_port=MASTER_PORT,
    )
    commands.append(run_command)

  utils.run_threads(vms, commands, login_shell=True)


if __name__ == "__main__":
  main()
