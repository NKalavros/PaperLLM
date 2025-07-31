import nemo.collections.asr as nemo_asr
import torch
import os
os.environ['OMP_NUM_THREADS'] = '2'
# Get the current number of threads
num_threads = torch.get_num_threads()
print(f"Current number of threads: {num_threads}")

# Set a custom number of threads (e.g., equal to the number of physical cores to avoid oversubscription)
# You might want to experiment with different values to find the optimal for your setup.
torch.set_num_threads(1)  # Or a lower value if experiencing oversubscription
torch.set_num_interop_threads(1)) # Or a lower value

import os
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
os.system('ffmpeg -i  complete_transcription/combined.mp3 -acodec pcm_s16le -ac 1 -y output.wav ')
output = asr_model.transcribe(['output.wav'])
#Write the output[0].text to a file
with open('Talk1.txt', 'w') as f:
    f.write(output[0].text)
# Convert to pdf with enscript in mac
os.remove('output.wav')  # Clean up the temporary file