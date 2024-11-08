#!/bin/bash

pushd env

podman build --ulimit nofile=$(ulimit -n):$(ulimit -n) -t lukasgd/ngc-jax:24.04-paxml .
enroot import -x mount -o /capstor/scratch/cscs/lukasd/images/ngc-jax+24.04-paxml.sqsh podman://lukasgd/ngc-jax:24.04-paxml

popd