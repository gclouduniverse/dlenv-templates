"""TensorFlow Multi-worker strategy entry point."""

import argparse

import json

import utils
import virtual_machine

_TF_PORT = 2222
SINGLE_QUOTE_ESCAPE = """'"'"'"""
DOUBLE_QUOTE_ESCAPE = '\\\"'


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
    default='example_trainers/tf_multiworker'
  )

  args = parser.parse_args()
  src = args.src

  vms = []
  for i in range(args.vm_num):
    vm_name = '{}-{}'.format(args.vm_name, str(i))
    vms.append(virtual_machine.VirtualMachine(vm_name))

  # prepare
  for vm in vms:
    utils.scp(src, vm.name, 'trainer', user=args.user)

  # run
  commands = []
  for i, _ in enumerate(vms):
    tf_config = {
      'cluster': {
        'worker': [f'{vm.internal_ip}:{_TF_PORT}' for vm in vms]
      },
      'task': {
        'type': 'worker',
        'index': i
      }
    }

    tf_config_str = json.dumps(tf_config, sort_keys=True)
    tf_config_env = (f"TF_CONFIG='{tf_config_str}'"
                     .replace('"', DOUBLE_QUOTE_ESCAPE)
                     .replace("'", SINGLE_QUOTE_ESCAPE))

    run_command = f'{tf_config_env} python trainer/task.py'
    commands.append(run_command)

  utils.run_threads(vms, commands, login_shell=True)


if __name__ == "__main__":
  main()
