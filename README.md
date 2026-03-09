- python train.py --config-name=codec_train

- for multi-node training, use torchrun to launch the training script.

the original flexicodec implementation is in zero_shot_tts_training/realtime_communication/taste_v2/modeling_dualcodec.py

However, it was used in the old server and might be used as a reference for you.
After I switched server, I lost access to the exact reproduction encironment (and data), and did not fully test out the new code. 
I recently briefly tested the code, and the resulting model seems to have lower acoustic quality than anticipated, but semantic quality is good. 
I'm not sure the cause for it, and maybe this is due to the mel spectrogram reconstruction settings

Tips for training flexicodec