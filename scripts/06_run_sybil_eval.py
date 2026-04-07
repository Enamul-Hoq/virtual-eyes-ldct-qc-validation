import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from virtual_eyes.downstream.sybil_eval import main

if __name__ == "__main__":
    print("Running Sybil evaluation...")
    main()
