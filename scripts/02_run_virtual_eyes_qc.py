import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from virtual_eyes.qc.run_qc import main

if __name__ == "__main__":
    print("Running Virtual-Eyes QC pipeline...")
    main()
    print("QC completed.")
