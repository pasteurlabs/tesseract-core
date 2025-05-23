#!/usr/bin/env bash
# vim: filetype=bash

set -euo pipefail

# Populate these values with your infrastructure
REPOSITORY="cr-tesseract-vms.azurecr.io"
RESOURCE_GROUP="rg-tesseract-vms"
VNET="vn-tesseract-vms"
SUBNET="sn-tesseract-vms"


error () {
	echo "$*"
	exit 1
}

USAGE="Usage: $0 [OPTION] TESSERACT

TESSERACT must be the Tesseract Image name without the repository, e.g.
\"meshstats:v1.0.0\".

--vm-size SIZE  defaults to Standard_F2s_v2 (2vCPUs, 4GiB RAM). Other options
                are described in
                https://azure.microsoft.com/en-gb/pricing/details/virtual-machines/linux/#pricing
                Standard_NC4as_T4_v3 is a size with 1 Nvidia T4 GPU.

--gpu           Add this flag to install Nvidia drivers and CUDA toolkit.

--ip-address IP Skip creating the VM and instead run the script in the given IP
                address.
"

[[ $# -gt 0 ]] || error "${USAGE}"

# Default values
baseos_image="Ubuntu2204"
vm_size="Standard_F2s_v2"


# Parse command line arguments/flags
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
	case $1 in
		--gpu)
			GPU=YES
			shift # past argument
			;;
		-i|--ip-address)
			ip="${2}"
			shift # past argument
			shift # past value
			;;
		-s|--vm-size)
			vm_size="${2}"
			shift # past argument
			shift # past value
			;;
		-h|--help)
			error "${USAGE}"
			;;
		*) # positional arg
			POSITIONAL_ARGS+=("${1}")
			shift
			;;
	esac
done
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

if [[ -z ${1+x} ]]; then
	 echo -e "Missing Tesseract Image\n"
	 error "${USAGE}"
fi

tesseract="${1}"
TESSERACT_NAME=$(echo "${tesseract}" | cut -d ':' -f 1)
ad_user=$(az ad signed-in-user show --query userPrincipalName \
          | tr -d '"' \
          | cut -f 1 -d '@')
vm_name="${ad_user}-tesseract-${TESSERACT_NAME}-${RANDOM}"

echo -e "Tagging Tesseract image"
tesseract_image="${REPOSITORY}/${ad_user}/${tesseract}"
docker image tag "${tesseract}" "${tesseract_image}"

echo -e "Pushing Tesseract image ${tesseract_image}"
docker push "${tesseract_image}"


if [[ -z ${ip+x} ]]; then
	echo -e "\n\nCreating VM ${vm_name} of instance type ${vm_size}"
	vm_info=$(az vm create --resource-group "${RESOURCE_GROUP}" \
			       --subnet "${SUBNET}" \
			       --vnet-name "${VNET}" \
			       --generate-ssh-keys \
			       --public-ip-address '' \
			       --enable-secure-boot false \
			       --output json \
			       --enable-auto-update true \
			       --admin-username "tessie" \
			       --name "${vm_name}" \
			       --image "${baseos_image}" \
			       --size "${vm_size}")

	ip=$(echo "${vm_info}" | jq -r .privateIpAddress)
else
	# shellcheck disable=SC2250
	vm_name=$(az vm list --resource-group "${RESOURCE_GROUP}" --show-details \
                  | jq --arg ip "$ip" '.[] | select(.privateIps == $ip) | .name')

	if [[ -z "${vm_name}" ]]; then
		error "\n\nVM with given IP address not found."
	fi

fi
echo -e "\n\nVM ${vm_name} ip address: ${ip}"


SSH="ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa tessie@${ip}"

if [[ "${GPU-}" == "YES" ]]; then
	echo -e "\n\nInstalling Nvidia drivers. This will take a while."

	# Instructions from
	# https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup

	${SSH} "sudo apt-get update"
	${SSH} "sudo apt-get install --yes ubuntu-drivers-common"
	${SSH} "sudo ubuntu-drivers install"

	${SSH} "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
	${SSH} "sudo apt-get install --yes ./cuda-keyring_1.1-1_all.deb"
	${SSH} "sudo apt-get update"
	${SSH} "sudo apt-get install --yes cuda-toolkit-12-5"

	DOCKER_GPU_FLAG="--gpus all"
	EXTRA_DOCKER_PACKAGE="nvidia-docker2"
fi


echo -e "\n\nSetting up Docker"
${SSH} 'sudo apt-get install --yes apt-transport-https ca-certificates curl \
                                   gnupg-agent software-properties-common'
${SSH} 'curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -'

# shellcheck disable=SC2016
${SSH} 'sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
                              $(lsb_release -cs) \
                              stable"'
${SSH} "sudo apt-get install --yes docker-ce docker-ce-cli containerd.io ${EXTRA_DOCKER_PACKAGE-}"

echo -e "\n\nRebooting VM" # Hard requirement for Nvidia/CUDA drivers, nice to have for Docker
set +e # The reboot command terminates the SSH session with non-zero status.
${SSH} "sudo reboot"
set -e

echo -e "\n\nWaiting for machine to boot."
sleep 1
until ${SSH} "exit" > /dev/null 2>&1; do
	sleep 1
done

docker_token=$(az acr login --name ${REPOSITORY/.azurecr.io/} --expose-token \
               | jq -r .accessToken)
${SSH} " echo ${docker_token} | sudo docker login ${REPOSITORY} -u 00000000-0000-0000-0000-000000000000 --password-stdin"

echo -e "\n\nStarting Tesseract server"
${SSH} "sudo docker run --restart always ${DOCKER_GPU_FLAG-} ${DOCKER_NFS_FLAG-} --detach --publish 8000:8000 ${tesseract_image} serve" || echo -e "Failed to start container, is it already running?"

echo -e "\n\nTesseract ${tesseract} deployed."
echo -e "Machine ${vm_name} IP is ${ip} and Tesseract is listening on port 8000."
echo -e "You can SSH into the machine using your local SSH key: \`${SSH}\`."

echo -e "\nNote: remember to destroy the VM after using it. "
echo -e "You can delete it via Azure Portal (also delete associated resources) or with"
echo -e "\`az vm delete --resource-group ${RESOURCE_GROUP} --name ${vm_name}\`"
