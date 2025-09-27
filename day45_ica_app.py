import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Streamlit App
st.title("Independent Component Analysis (ICA) Demo")
st.write("""
🎵 ICA is a technique to separate mixed signals into independent sources.  
Here we demonstrate with synthetic signals.
""")

# Generate synthetic signals
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)          # Signal 1 : Sinusoidal
s2 = np.sign(np.sin(3 * time)) # Signal 2 : Square wave
s3 = np.random.normal(size=n_samples) # Signal 3 : Random noise

S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # Standardize

# Mixing matrix (random)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T  # Mixed signals

# ICA
ica = FastICA(n_components=3, random_state=42)
S_ica = ica.fit_transform(X)  # Recovered signals
A_ica = ica.mixing_

# Show plots
fig, axes = plt.subplots(3, 1, figsize=(10, 7))
axes[0].set_title("Original Signals")
for sig in S.T:
    axes[0].plot(sig)

axes[1].set_title("Mixed Signals")
for sig in X.T:
    axes[1].plot(sig)

axes[2].set_title("Recovered Signals (ICA)")
for sig in S_ica.T:
    axes[2].plot(sig)

plt.tight_layout()
st.pyplot(fig)

st.success("✅ ICA successfully separated the signals!")
