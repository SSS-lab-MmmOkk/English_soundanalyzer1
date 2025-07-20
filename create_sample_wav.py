import soundfile as sf
import numpy as np

# Create a silent 5-second audio file
sr = 22050  # sample rate
duration = 5
y = np.zeros(sr * duration)

sf.write('sample.wav', y, sr)
