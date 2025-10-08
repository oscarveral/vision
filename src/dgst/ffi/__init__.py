"""Foreign Function Interface for filters."""

makefile_dir = os.path.dirname(__file__)

makefile_path = os.path.join(makefile_dir, 'Makefile')
print(f"Makefile path: {makefile_path}")
if os.path.exists(makefile_path):
    try:
        subprocess.run(['make', '--quiet', '-C', makefile_dir], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Makefile execution failed: {e}")
else:
    raise FileNotFoundError(f"Makefile not found in {makefile_dir}")