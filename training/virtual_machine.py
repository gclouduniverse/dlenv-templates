"""Virtual machine class."""
import os

import utils


class VirtualMachine:
  """GCP virtual machine."""

  def __init__(self, name):
    self.name = name
    self.internal_ip = self.get_vm_internal_ip()

  def run_remote_command(self, command, login_shell=False):
    """Run remote command."""
    if login_shell:
      command = 'bash -l -c "{}"'.format(command)
    remote_command = utils.SSH_COMMAND.format(self.name, command)
    print('Running command: %s' % remote_command)
    os.system(remote_command)

  def authenticate_vm(self, private_key_path, user):
    """Authenticate the VM to access all peers."""
    utils.scp(private_key_path, self.name, '~/.ssh/id_rsa', user=user)
    self.run_remote_command(
      'echo "Host *\n  StrictHostKeyChecking no\n" > ~/.ssh/config')
    self.run_remote_command('chmod 400 ~/.ssh/config')

  def get_vm_internal_ip(self, zone='us-west1-b'):
    """Get VM's internal IP."""
    command = utils.COMMAND_GET_IP.format(self.name, zone)
    return os.popen(command).read().rstrip()
