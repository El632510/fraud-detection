# tests/test_basic.py
# Basic sanity tests for the fraud detection project

def test_imports():
    """Test that core libraries are importable"""
    import pandas as pd
    import numpy as np
    import sklearn
    assert True

def test_pandas_version():
    """Test pandas is available"""
    import pandas as pd
    assert pd.__version__ is not None

def test_numpy_version():
    """Test numpy is available"""
    import numpy as np
    assert np.__version__ is not None