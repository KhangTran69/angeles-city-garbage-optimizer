import sys
import os
import streamlit as st
st.write(f"Python executable: {sys.executable}")
st.write(f"sys.path: {sys.path}")
try:
    import ortools
    st.write(f"ortools version: {ortools.__version__}")
except ImportError as e:
    st.write(f"Failed to import ortools: {e}")
