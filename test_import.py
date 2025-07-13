#!/usr/bin/env python3

print("Testing library imports...")

try:
    import pandas as pd
    print("✅ pandas imported successfully")
    print(f"   Version: {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas import failed: {e}")

try:
    import sklearn
    print("✅ scikit-learn imported successfully")
    print(f"   Version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")

try:
    import streamlit as st
    print("✅ streamlit imported successfully")
    print(f"   Version: {st.__version__}")
except ImportError as e:
    print(f"❌ streamlit import failed: {e}")

try:
    import seaborn as sns
    print("✅ seaborn imported successfully")
    print(f"   Version: {sns.__version__}")
except ImportError as e:
    print(f"❌ seaborn import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported successfully")
    import matplotlib
    print(f"   Version: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")

print("\n" + "="*50)
print("Library import test completed!")
print("="*50)