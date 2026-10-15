#!/usr/bin/env bash
set -euo pipefail

FOVEAVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${FOVEAVLA_ROOT}"
REPOSITORY="${FAST_FOUNDATION_REPOSITORY:-${FOVEAVLA_ROOT}/.runtime/Fast-FoundationStereo}"
CHECKPOINT="${FAST_FOUNDATION_CHECKPOINT:-${REPOSITORY}/weights/23-36-37/model_best_bp2_serialize.pth}"
COMMIT=a290ba04c1b3ad1ec41a33974a157b2917b624d4
SHA256=7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214

if [[ ! -e "${REPOSITORY}" ]]; then
    git clone https://github.com/NVlabs/Fast-FoundationStereo.git "${REPOSITORY}"
    git -C "${REPOSITORY}" checkout "${COMMIT}"
fi
if [[ "$(git -C "${REPOSITORY}" rev-parse HEAD)" != "${COMMIT}" || \
      -n "$(git -C "${REPOSITORY}" status --porcelain --untracked-files=no)" ]]; then
    echo "Fast-FoundationStereo must be a clean checkout of ${COMMIT}: ${REPOSITORY}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    mkdir -p "$(dirname "${CHECKPOINT}")"
    curl -fL --retry 3 \
        https://huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/model_best_bp2_serialize.pth \
        -o "${CHECKPOINT}.partial"
    echo "${SHA256}  ${CHECKPOINT}.partial" | sha256sum --check
    mv "${CHECKPOINT}.partial" "${CHECKPOINT}"
fi
echo "${SHA256}  ${CHECKPOINT}" | sha256sum --check
echo "Fast-FoundationStereo source and weights are ready. See docs/deployment.md."
