#!/usr/bin/env bash
set -euo pipefail

FOVEAVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${DROID_PYTHON:-python}"
export LD_LIBRARY_PATH="/usr/local/zed/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Reuse the configured DROID environment, as in OpenPI's DROID example.
"${PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit("Activate your configured Python 3.10 DROID environment first.")
PY
DROID_COMMIT=33ae6a67274f36d2e29525b86f23a56616ef43a7
DROID_ROOT="$(cd "${DROID_ROOT:-${FOVEAVLA_ROOT}/.runtime/droid}" 2>/dev/null && pwd)" || {
    echo "Clone DROID into .runtime/droid or set DROID_ROOT; see docs/deployment.md." >&2
    exit 1
}
if [[ "$(git -C "${DROID_ROOT}" rev-parse HEAD)" != "${DROID_COMMIT}" ]]; then
    echo "DROID must be checked out at ${DROID_COMMIT}: ${DROID_ROOT}" >&2
    exit 1
fi
IMPORTED_ROOT="$("${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("droid")
if spec is None or spec.origin is None:
    raise SystemExit("Install and configure DROID before installing the FoveaVLA client.")
print(Path(spec.origin).resolve().parents[1])
PY
)"
if [[ "${IMPORTED_ROOT}" != "${DROID_ROOT}" ]]; then
    echo "The active environment imports DROID from ${IMPORTED_ROOT}, not ${DROID_ROOT}." >&2
    exit 1
fi
echo "Patching DROID at ${DROID_ROOT}"

# DROID pins opencv-python 4.6, which needs NumPy 1.x. pyzed 5.1 declares numpy>=2 but runs with 1.26.
"${PYTHON}" -m pip install -e "${FOVEAVLA_ROOT}/packages/foveavla-client" tyro "numpy<2"

# Apply only the changed startup/camera lines, preserving hardware configuration.
for patch in droid_server_interface_startup droid_zed_grab_error; do
    patch_path="${FOVEAVLA_ROOT}/scripts/patches/${patch}.patch"
    if git -C "${DROID_ROOT}" apply --unidiff-zero --reverse --check "${patch_path}" 2>/dev/null; then
        echo "${patch}: already applied"
    elif git -C "${DROID_ROOT}" apply --unidiff-zero --check "${patch_path}"; then
        git -C "${DROID_ROOT}" apply --unidiff-zero "${patch_path}"
    else
        echo "Patch does not match your DROID version: ${patch_path}" >&2
        exit 1
    fi
done

"${PYTHON}" - <<'PY'
from droid.misc.parameters import nuc_ip
import pyzed.sl as sl
if not nuc_ip:
    raise SystemExit("Set nuc_ip in your DROID droid/misc/parameters.py before deployment.")
if sl.Camera.get_sdk_version() != "5.1.2":
    raise SystemExit("FoveaVLA requires ZED SDK 5.1.2 and its Python 3.10 API.")
print("FoveaVLA client ready in the existing DROID environment. No robot was started.")
PY
