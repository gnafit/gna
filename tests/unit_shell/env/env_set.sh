#!/usr/bin/env bash

./gna \
    -- env-set -r test0 '{key1: string, key2: 1.0}' \
    -- env-print test0 \
    -- env-set -r test1 -y sub '{key1: string, key2: 1.0}' \
    -- env-print test1 \
    -- env-set -r test2 -a key1 string \
    -- env-print test2
