import nemo.collections.asr as nemo_asr
import torch
os.environ['OMP_NUM_THREADS'] = '2'
# Get the current number of threads
num_threads = torch.get_num_threads()
print(f"Current number of threads: {num_threads}")

# Set a custom number of threads (e.g., equal to the number of physical cores to avoid oversubscription)
# You might want to experiment with different values to find the optimal for your setup.
torch.set_num_threads(2)  # Or a lower value if experiencing oversubscription
torch.set_num_interop_threads(2) # Or a lower value

import os
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
os.system('ffmpeg -i HITS_JSR.mp3 -acodec pcm_s16le -ac 1 output.wav ')
output = asr_model.transcribe(['/Users/nikolas/Downloads/output.wav'])
