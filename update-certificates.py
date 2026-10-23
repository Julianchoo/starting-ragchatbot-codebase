"""Update Python's certificate tools. Run with the Python used by this project."""
import importlib.util
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Use Python 3.10 or later to install the corrected certificate tools.")

if importlib.util.find_spec("pip") is None:
    subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])

# The official wheel hash protects the bootstrap even when the old pip trusts bad CAs.
PIP_WHEEL = (
    "https://files.pythonhosted.org/packages/3f/06/"
    "3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/"
    "pip-26.2.2-py3-none-any.whl"
    "#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", PIP_WHEEL])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "certifi>=2026.10.19", "uv>=0.13.4"])
