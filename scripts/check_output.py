import pickle
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python check_output.py <filepath>")
    sys.exit(1)

dict_file = Path(sys.argv[1])

if dict_file.exists():
    print(f"Loading dictionary from: {dict_file}")
    
    # Open and load the dictionary
    with open(dict_file, 'rb') as f:
        data = pickle.load(f)
    
    # Dump everything to console
    print("\n" + "="*50)
    print("Dictionary Contents:")
    print("="*50 + "\n")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        print(data)
    print("\n" + "="*50)
else:
    print(f"File not found: {dict_file}")
