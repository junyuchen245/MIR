#!/usr/bin/env bash

bash ./build.sh

docker save vfa_lumir25 | gzip -c > vfa_lumir25.tar.gz
