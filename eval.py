import os
import librosa
import librosa.display
import pretty_midi
import numpy as np
import subprocess

from pathlib import Path
from music21 import converter, midi
from mir_eval.transcription import precision_recall_f1_overlap, onset_precision_recall_f1, offset_precision_recall_f1

# MusicXML to MIDI
def musicxml_to_midi(musicxml_file):
    score = converter.parse(musicxml_file)
    midi_stream = score.write('midi')
    print(f"MIDI file saved as {midi_stream}")
    return midi_stream
    

# Align MIDI using DTW
def extract_notes(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.pitch))
    
    notes.sort(key=lambda x: x[0])
    return np.array(notes)


def align_midi_with_dtw(midi_file_1, midi_file_2):
    notes_1 = extract_notes(midi_file_1)
    notes_2 = extract_notes(midi_file_2)
    
    time_1 = notes_1[:, 0]
    pitch_1 = notes_1[:, 1]
    
    time_2 = notes_2[:, 0]
    pitch_2 = notes_2[:, 1]
    
    # Convert time-pitch sequences into 2D feature matrices
    # Feature vector: [time, pitch]
    features_1 = np.vstack((time_1, pitch_1))
    features_2 = np.vstack((time_2, pitch_2))

    # Perform DTW using librosa
    D, wp = librosa.sequence.dtw(X=features_1, Y=features_2)
    
    return D, wp
    

def audio_to_midi(data_folder_path, audio_files_list):
    """
        Given a list of audio paths, generates transcribed midi files.

        data_folder_path: path to the data folder , e.g. /mnt/sdb/ibukey/asap-dataset
        audio_files_list: audio file paths, e.g. Bach/Fugue/bwv_848/Denisova06M.wav (so full path becomes data_folder_path/audio_file_path)
    """
    # Create file list for docker
    transcribed_midi_files = []
    datafolder = os.path.basename(data_folder_path)
    with open("file_list.txt", "w") as file:
        for audio in audio_files_list:
            file.write(f'/opt/{datafolder}/{audio}\n')
            transcribed_midi_files.append(f'/{os.getcwd()}/midi/{datafolder}/{audio}.midi')

    # Transcribe
    os.makedirs('midi', exist_ok=True)
    subprocess.run(['docker', 'build', '-t', 'onf', '.'], check=True)
    subprocess.run(['docker', 'run', '-v', f'{os.getcwd()}/midi:/opt/midi', '-v', f'{data_folder_path}:/opt/{datafolder}', '-t', 'onf'], check=True)

    return transcribed_midi_files

    # Transcriptions will be written inside midi/ folder

def eval_midi(reference_midi, generated_midi, metric="precision_recall_f1_overlap"):

    D, wp = align_midi_with_dtw(reference_midi, generated_midi)

    midi_data = pretty_midi.PrettyMIDI(reference_midi)
    ref_intervals, ref_pitches = [], []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            ref_intervals.append((note.start, note.end))
            ref_pitches.append(note.pitch)
    ref_intervals = np.array(ref_intervals)
    ref_pitches = np.array(ref_pitches)


    midi_data = pretty_midi.PrettyMIDI(generated_midi)
    est_intervals, est_pitches = [], []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            est_intervals.append((note.start, note.end))
            est_pitches.append(note.pitch)
    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)
    
    ref_intervals = ref_intervals[wp[::-1].T[0]]
    ref_pitches = ref_pitches[wp[::-1].T[0]]
    est_intervals = est_intervals[wp[::-1].T[1]]
    est_pitches = est_pitches[wp[::-1].T[1]]

    if metric == "precision_recall_f1_overlap":
        precision, recall, f_measure, avg_overlap_ratio = precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches)
    elif metric == "onset_precision_recall_f1":
        precision, recall, f_measure = onset_precision_recall_f1(ref_intervals, est_intervals)
        avg_overlap_ratio = None
    elif metric == "offset_precision_recall_f1":
        precision, recall, f_measure = offset_precision_recall_f1(ref_intervals, est_intervals)
        avg_overlap_ratio = None
    else:
        raise NotImplementedError(f"{metric} is not a valid metric.")
    
    return precision, recall, f_measure, avg_overlap_ratio


def eval_audio(data_folder_path, reference_audio, generated_audio, metric="precision_recall_f1_overlap"):

    midifiles = audio_to_midi(data_folder_path, [reference_audio, generated_audio])
    
    reference_midi = midifiles[0]
    generated_midi = midifiles[1]

    return eval_midi(reference_midi, generated_midi, metric=metric)    
    