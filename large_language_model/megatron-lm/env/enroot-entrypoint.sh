#!/usr/bin/env bash


mpi_local_rank=${SLURM_LOCALID:-0}
enroot_ssh_port=${ENROOT_SSH_PORT:-15263}

if [ "${mpi_local_rank}" -eq 0 ]; then
    ln -s /bin/bash /usr/local/bin/bash
    echo "[enroot-entrypoint] Launching SSH-server on $(hostname) at port ${enroot_ssh_port}"
    sed -i -e "s/#Port 22/Port ${enroot_ssh_port}/g" /etc/ssh/sshd_config
    /usr/sbin/sshd
    service ssh status
fi

exec "$@"