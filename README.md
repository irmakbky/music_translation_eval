# Multimodal Music Translation Evaluation

The eval code uses Docker to transcribe audio to MIDI using the Onsets and Frames transcription model. But currently, the rest of the code is executable via python files or jupyter notebook.

Relevant functions in the **eval.py** file:
- **audio_to_midi**: Takes in a data folder path (the folder where audio files are stored) and a list of audio file paths in that folder, and generates transcribed MIDIs for each audio file. (The reason folder path and audio files paths are passed in separately is due to folder naming issues in Docker. Refer to docstring for this function in eval.py for examples)
- **eval_midi**: Takes in two MIDI files, aligns them using DTW and runs evaluation metrics.
- **eval_audio**: Takes in two audio files, transcribes them into MIDI, and passes them into eval_midi.
