#!/bin/bash

while IFS= read -r wav_file; do
    echo "Processing: $wav_file"
    PIECE=$wav_file

    python /opt/magenta/magenta/models/onsets_frames_transcription/onsets_frames_transcription_transcribe.py \
        --model_dir="/opt/train" "$wav_file"
done < /opt/file_list.txt
