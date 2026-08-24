import sys

# diffusers' scheduler import chain hits scipy's lazy submodule loader
# (scipy/__init__.py's __getattr__) before scipy is fully initialized,
# which recurses back into itself — "maximum recursion depth exceeded"
# with no other cause in the traceback. Importing the submodules eagerly,
# ahead of anything that would trigger the lazy path, sidesteps it; the
# raised limit is a harmless second line of defense either way.
sys.setrecursionlimit(5000)
import scipy  # noqa: E402
import scipy.integrate  # noqa: E402
import scipy.special  # noqa: E402

import runpod  # noqa: E402

from runpod_inference import handle_job  # noqa: E402


def handler(job):
    return handle_job(job)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
