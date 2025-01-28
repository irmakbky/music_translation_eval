import os
import scipy
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
    
    # Perform DTW using librosa
    D, wp = librosa.sequence.dtw(X=notes_1[:, 1], Y=notes_2[:, 1])
    
    return D, wp
    

def audio_to_midi(data_folder_path, audio_files_list):
    """
        Given a list of audio paths, generates transcribed midi files.

        data_folder_path: path to the data folder , e.g. /mnt/sdb/ibukey/asap-dataset
        audio_files_list: audio file paths, e.g. Bach/Fugue/bwv_848/Denisova06M.wav (so full path becomes data_folder_path/audio_file_path)
    """
    # Create file list for docker
    transcribed_midi_files, onf_score_files = [], []
    datafolder = os.path.basename(data_folder_path)
    with open("file_list.txt", "w") as file:
        for audio in audio_files_list:
            file.write(f'/opt/{datafolder}/{audio}\n')
            transcribed_midi_files.append(f'/{os.getcwd()}/midi/{datafolder}/{audio}.midi')
            onf_score_files.append(f'/{os.getcwd()}/onf_score/{datafolder}/{audio[:-4]}_preds.npy')

    # Transcribe
    os.makedirs('midi', exist_ok=True)
    os.makedirs('onf_score', exist_ok=True)
    subprocess.run(['docker', 'build', '-t', 'onf', '.'], check=True)
    subprocess.run(['docker', 'run', '-v', f'{os.getcwd()}/midi:/opt/midi', '-v', f'{data_folder_path}:/opt/{datafolder}', '-v', f'{os.getcwd()}/onf_score:/opt/onf_score', '-t', 'onf'], check=True)

    return transcribed_midi_files, onf_score_files

    # Transcriptions will be written inside midi/ folder, O&F predictions will be written in onf_score/ folder 

def eval_midi(reference_midi, generated_midi, reference_frames, generated_frames, metric="precision_recall_f1_overlap"):

    D, wp = librosa.sequence.dtw(reference_frames, generated_frames)
    seen = set()
    new_wp = np.array([(a, b) for a, b in wp[::-1] if b not in seen and not seen.add(b)])
    interp_func = scipy.interpolate.interp1d(new_wp[:, 1], new_wp[:, 0], kind='linear', fill_value="extrapolate")

    midi_data = pretty_midi.PrettyMIDI(reference_midi)
    ref_intervals, ref_pitches = [], []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            ref_intervals.append((note.start, note.end))
            ref_pitches.append(note.pitch)
    ref_intervals = np.array(ref_intervals)
    ref_pitches = np.array(ref_pitches)

    frame_rate = 16000/512
    midi_data = pretty_midi.PrettyMIDI(generated_midi)
    est_intervals, est_pitches = [], []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = interp_func(note.start * frame_rate).item() / frame_rate
            end = interp_func(note.end * frame_rate).item() / frame_rate
            if start == end: # if interpolation causes start and end to be the same due to short duration
                end += 1e-9
            est_intervals.append((start, end))
            est_pitches.append(note.pitch)
    est_intervals = np.array(est_intervals)
    est_pitches = np.array(est_pitches)

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

    midifiles, onf_score_files = audio_to_midi(data_folder_path, [reference_audio, generated_audio])
    
    reference_midi = midifiles[0]
    generated_midi = midifiles[1]

    # get frame predictions
    reference_frames = np.load(onf_score_files[0], allow_pickle=True)[0]['frame_predictions'][0].T
    generated_frames = np.load(onf_score_files[1], allow_pickle=True)[0]['frame_predictions'][0].T

    return eval_midi(reference_midi, generated_midi, reference_frames, generated_frames, metric=metric)    
    