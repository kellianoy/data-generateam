# How to launch the docker container to test the program

On Windows 10, to launch the docker container, you need to install the following softwares:

- Git Bash
- WSL2
- Docker
  
## Install Git Bash

Basically, just get it from [here](https://git-scm.com/downloads).

## Install WSL2

Install WSL following [this](https://docs.microsoft.com/en-us/windows/wsl/install-win10) tutorial.

Set the default version to WSL2 by doing `wsl --set-default-version 2` in a powershell.

Then, download Ubuntu 22.04 LTS from the windows store. Once it is downloaded, launch it and follow the instructions to install it.

Select Ubuntu 22.04 LTS as your default distribution by doing `wsl --set-default Ubuntu-22.04` in a powershell.

## Install Docker

Get it from [here](https://docs.docker.com/docker-for-windows/install/). Then, activate the WSL2 backend in the parameters.

## Verification

When launching your Ubuntu WSL, check you have access to git by doing `git --version`. Then, check you have access to docker by doing `docker --version`.

Once it is done, you can simply open your WSL in the folder of the project by doing `shift + right click` and selecting the **shell interpreter** (not the powershell). Then, do `sh run.sh` to launch the docker container.
