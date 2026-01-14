# eplus/colab_bootstrap.py
import os, sys, subprocess, shlex, pathlib

EPLUS_URL = "https://github.com/NREL/EnergyPlus/releases/download/v25.1.0/EnergyPlus-25.1.0-68a4a7c774-Linux-CentOS7.9.2009-x86_64.tar.gz"
EPLUS_DEST = str(pathlib.Path.home() / "EnergyPlus-25-1-0")
LIB_PATH = "/usr/lib/x86_64-linux-gnu"  # for libssl1.1

APT_BASE_PKGS = [
    "libxkbcommon-x11-0",
    # optional/headless helpers
    "libxcb-xinerama0", "libx11-xcb1", "libnss3", "libxi6", "libxtst6",
    "libxcb-icccm4", "libxcb-image0", "libxcb-keysyms1",
    "libxcb-render0", "libxcb-shape0", "libxcb-xfixes0", "libxcb-randr0",
    "libxrender1", "libasound2", "libxcb-render-util0",
]

UBUNTU_OPENSSL_11_VERS = [
    "1.1.1f-1ubuntu2.22", "1.1.1f-1ubuntu2.21",
    "1.1.1f-1ubuntu2.20", "1.1.1f-1ubuntu2.19", "1.1.1f-1ubuntu2",
]

def _run(cmd, quiet=True, check=True, shell=False, env=None, cwd=None):
    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    kwargs = dict(check=check, cwd=cwd, env=env)
    if quiet:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return subprocess.run(cmd, **kwargs)

def _apt_install(pkgs):
    _run("apt-get update -y")
    # tolerate missing optional libs
    _run(["apt-get", "install", "-y", *pkgs], check=False)

def _ensure_libssl11():
    # Try install libssl1.1 from Ubuntu security archive (as in your snippet)
    for v in UBUNTU_OPENSSL_11_VERS:
        url = f"http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_{v}_amd64.deb"
        deb = "/tmp/libssl1.1.deb"
        if _run(["wget", "-q", url, "-O", deb], check=False).returncode == 0:
            _run(["apt-get", "install", "-y", deb], check=True)
            break

def _download_energyplus():
    # Idempotent: skip if already extracted
    if pathlib.Path(EPLUS_DEST, "energyplus").exists():
        return
    tar = "/tmp/eplus.tar.gz"
    _run(["rm", "-f", tar])
    pathlib.Path(EPLUS_DEST).mkdir(parents=True, exist_ok=True)
    _run(["wget", "-q", "--https-only", "--retry-connrefused", "--tries=3", EPLUS_URL, "-O", tar])
    _run(["tar", "-xzf", tar, "-C", EPLUS_DEST, "--strip-components=1"])

def _set_env_for_current_process():
    os.environ["ENERGYPLUSDIR"] = EPLUS_DEST
    os.environ["LD_LIBRARY_PATH"] = f"{EPLUS_DEST}:{LIB_PATH}:" + os.environ.get("LD_LIBRARY_PATH", "")
    if EPLUS_DEST not in sys.path:
        sys.path.insert(0, EPLUS_DEST)

def prepare_colab_eplus(silent: bool = True) -> None:
    """
    Perform Colab runtime prep silently (no prints). Raises on hard failures.
    Steps:
      - apt-get base libs
      - install libssl1.1
      - download & extract EnergyPlus 25.1.0
      - set ENERGYPLUSDIR, LD_LIBRARY_PATH, and sys.path in this process
    """
    # base libs
    _apt_install(APT_BASE_PKGS)
    # libssl1.1 shim
    _ensure_libssl11()
    # EnergyPlus bits
    _download_energyplus()
    # env for *this* Python process
    _set_env_for_current_process()
    # quick smoke (silent; raises if fails)
    _run([f"{EPLUS_DEST}/energyplus", "--version"])

def main():
    # CLI entry point: --verbose optional
    verbose = "--verbose" in sys.argv
    try:
        prepare_colab_eplus(silent=not verbose)
    except Exception as e:
        # keep stderr clean unless verbose; still exit nonzero on failure
        if verbose:
            print(f"[one-client] bootstrap failed: {e}", file=sys.stderr)
        raise

# Optional: allow `python -m one_client.colab_bootstrap`
if __name__ == "__main__":
    main()
