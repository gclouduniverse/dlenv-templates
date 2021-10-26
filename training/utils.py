"""Util functions."""
import os
import threading

COMMAND_GET_IP = ("gcloud compute instances describe {} --zone={}" +
                  " --format='get(networkInterfaces[0].networkIP)'")

SSH_COMMAND = "gcloud compute ssh {} --ssh-flag='-t' --command='{}'"

SCP_COMMAND = "gcloud compute scp --recurse {src} {address}:{destination}"


def run_threads(vms, commands, login_shell=False):
  """Run commands in parallel threads."""
  threads = []
  for vm, command in zip(vms, commands):
    if login_shell:
      command = 'bash -l -c "{}"'.format(command)
    ssh_command = SSH_COMMAND.format(vm.name, command)
    print('Running command: %s' % ssh_command)
    threads.append(threading.Thread(target=os.system, args=(ssh_command,)))

  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()


def scp(src, vm_name, destination, user=None):
  """SCP file to VM."""
  if not user:
    address = vm_name
  else:
    address = '{}@{}'.format(user, vm_name)

  scp_command = SCP_COMMAND.format(
    src=src,
    address=address,
    destination=destination
  )
  os.system(scp_command)
