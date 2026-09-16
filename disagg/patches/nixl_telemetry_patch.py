"""
NIXL compatibility patch for NIXL v1.3.0 + vLLM 0.23.0.

Fix 1: NIXL _api.py get_xfer_telemetry → return None instead of raising
Fix 2: vLLM worker.py record_transfer → guard against None result
"""
import os, re

NIXL_API = None
for p in [
    "/opt/nvidia/nvda_nixl/lib/python3/dist-packages/nixl/_api.py",
    "/opt/venv/lib/python3.12/site-packages/nixl/_api.py",
]:
    if os.path.exists(p):
        NIXL_API = p
        break

if not NIXL_API:
    print("WARNING: nixl _api.py not found — skipping")
    exit(0)

src = open(NIXL_API).read()

# Fix 1: get_xfer_telemetry returns None on NO_TELEMETRY
if "NO_TELEMETRY_PATCHED" not in src:
    OLD = re.compile(
        r'(    def get_xfer_telemetry\([^)]*\)[^:]*:\n)'
        r'(        return self\.agent\.getXferTelemetry\(handle\._handle\))'
    )
    NEW = (r'\1'
           r'        try:  # NO_TELEMETRY_PATCHED\n'
           r'            return self.agent.getXferTelemetry(handle._handle)\n'
           r'        except Exception as _e:\n'
           r'            if "NO_TELEMETRY" in str(_e) or "NoTelemetry" in type(_e).__name__:\n'
           r'                return None\n'
           r'            raise')
    patched, n = OLD.subn(NEW, src)
    if n:
        src = patched
        print("Fix 1 applied: get_xfer_telemetry returns None on NO_TELEMETRY")
    else:
        print("Fix 1: pattern not matched")
else:
    print("Fix 1 already applied")

# Remove any broken device_list patch if present
if "DEVICE_LIST_PATCHED" in src:
    src = re.sub(
        r'                # DEVICE_LIST_PATCHED.*?(?=                if nixl_conf\.num_threads)',
        '',
        src, flags=re.DOTALL
    )
    print("Removed broken device_list patch")

open(NIXL_API, "w").write(src)

# Fix 2: vLLM worker.py record_transfer guard
WORKER = "/tmp/vllm-base/vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py"
if os.path.exists(WORKER):
    wsrc = open(WORKER).read()
    if "TELEMETRY_GUARD_PATCHED" not in wsrc:
        OLD2 = "                        self.xfer_stats.record_transfer(res)"
        NEW2 = ("                        if res is not None:  # TELEMETRY_GUARD_PATCHED\n"
                "                            self.xfer_stats.record_transfer(res)")
        if OLD2 in wsrc:
            open(WORKER, "w").write(wsrc.replace(OLD2, NEW2, 1))
            print("Fix 2 applied: record_transfer guarded against None")
        else:
            print("Fix 2 already applied")
    else:
        print("Fix 2 already applied")

print("Done")
