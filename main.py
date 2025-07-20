import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def analyze_audio(file_path):
    """
    Analyzes an audio file to extract pitch, volume, and silent parts.

    Args:
        file_path (str): The path to the audio file (MP3 or WAV).

    Returns:
        tuple: A tuple containing:
            - times (np.ndarray): The time array for the x-axis.
            - pitch (np.ndarray): The extracted pitch (F0) values.
            - rms (np.ndarray): The extracted RMS (volume) values.
            - silent_intervals (list): A list of tuples with start and end times of silent parts.
    """
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract pitch (F0) using YIN
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Extract RMS (volume)
    rms = librosa.feature.rms(y=y)[0]

    # Detect silent intervals
    silent_intervals = librosa.effects.split(y, top_db=30)

    # Create time axis
    times = librosa.times_like(rms)

    return times, pitch, rms, silent_intervals

def visualize_audio_features(times, pitch, rms, silent_intervals, output_path='audio_analysis.png'):
    """
    Visualizes the extracted audio features.

    Args:
        times (np.ndarray): The time array for the x-axis.
        pitch (np.ndarray): The extracted pitch (F0) values.
        rms (np.ndarray): The extracted RMS (volume) values.
        silent_intervals (list): A list of tuples with start and end times of silent parts.
        output_path (str): The path to save the output plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot RMS (volume)
    plt.subplot(2, 1, 1)
    plt.plot(times, rms, label='Volume (RMS)')
    plt.title('Volume and Pitch Analysis')
    plt.ylabel('RMS')
    plt.legend()

    # Plot Pitch
    plt.subplot(2, 1, 2)
    plt.plot(times, pitch, label='Pitch (F0)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch (Hz)')
    plt.legend()

    # Highlight silent intervals
    for interval in silent_intervals:
        plt.axvspan(interval[0] / len(rms) * times[-1], interval[1] / len(rms) * times[-1], color='gray', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == '__main__':
    # This is a placeholder for a file path.
    # In a real application, you would get this from a file upload dialog.
    audio_file = 'sample.wav'

    try:
        times, pitch, rms, silent_intervals = analyze_audio(audio_file)
        visualize_audio_features(times, pitch, rms, silent_intervals)
        print(f"Analysis complete. Plot saved to audio_analysis.png")
    except FileNotFoundError:
        print(f"Error: The file '{audio_file}' was not found.")
        print("Please make sure a 'sample.wav' file exists in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
