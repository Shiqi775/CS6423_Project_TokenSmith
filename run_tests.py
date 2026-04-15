"""Run pytest and write results to a UTF-8 file, bypassing conda's GBK stdout."""
import subprocess, sys, pathlib

out_file = pathlib.Path("test_results.txt")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_hot_chunk_cache.py", "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
)
output = result.stdout + result.stderr
out_file.write_text(output, encoding="utf-8")
print(f"Exit code: {result.returncode}")
print(f"Results written to {out_file.resolve()}")
# Print last 30 lines as summary
lines = output.splitlines()
for line in lines[-30:]:
    print(line)
