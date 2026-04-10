#!/usr/bin/env bash

set -euo pipefail

bench_name="cpu_backend"
criterion_args=("$@")

echo "== Parallel build baseline =="
cargo bench --bench "$bench_name" -- --save-baseline parallel "${criterion_args[@]}"

echo
echo "== Serial build compared to parallel baseline =="
cargo bench --no-default-features --bench "$bench_name" -- --baseline parallel "${criterion_args[@]}"

echo
echo "Benchmarks completed. Criterion reports were written under target/criterion/."