"""TensorFlow Horovod entry point."""

import argparse

import utils
import virtual_machine


def main():
  parser = argparse.ArgumentParser(description='Optional app description')
  parser.add_argument('--vm-name', dest='vm_name', type=str, required=True)
  parser.add_argument('--vm-number', dest='vm_num', type=int, default=2)
  parser.add_argument('--gpu-per-vm', dest='gpu_per_vm', type=int, default=1)
  parser.add_argument('--private-key-path', dest='private_key_path', type=str)
  parser.add_argument('--user', dest='user', type=str)
  parser.add_argument(
    '--src',
    dest='src',
    type=str,
    default='example_trainers/tf_horovod'
  )

  args = parser.parse_args()
  gpu_per_vm = args.gpu_per_vm
  vm_num = args.vm_num
  total_gpu = gpu_per_vm * vm_num
  src = args.src

  vms = []
  for i in range(args.vm_num):
    vm_name = '{}-{}'.format(args.vm_name, str(i))
    vms.append(virtual_machine.VirtualMachine(vm_name))

  # authenticate the master VM
  vms[0].authenticate_vm(args.private_key_path, args.user)

  # prepare Horovod
  for vm in vms:
    utils.scp(src, vm.name, 'trainer', user=args.user)

  # run Horovod on master
  hosts = ','.join(
    ['{}:{}'.format(vm.name, gpu_per_vm) for _, vm in enumerate(vms)])

  run_command = ('horovodrun '
                 '-np {total_gpu} '
                 '-H {hosts} '
                 'python3 {script}').format(
    total_gpu=total_gpu,
    hosts=hosts,
    script='trainer/task.py'
  )
  vms[0].run_remote_command(run_command, login_shell=True)


if __name__ == "__main__":
  main()
