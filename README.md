# FlexiCodec Training Experimental Implementation

This is an experimental training implementation of flexicodec (inference script is https://github.com/amphionspace/FlexiCodec). 

The original flexicodec training implementation is in `zero_shot_tts_training/realtime_communication/taste_v2/modeling_dualcodec.py`


To proceed with this codebase: 
0. Download the Emilia data from huggingface. No need to unzip the tar files. Then, change conf/data/emiliawebdataset_static_audio.yaml to point to the downloaded data.
1. Replace the resume_path in `conf/model/bdcodec/flexicodec_reproduce.yaml` with the path to the downloaded https://huggingface.co/jiaqili3/flexicodec/blob/main/dualcodec_with_sensevoice_12hz_soundstream.safetensors.
when training flexicodec, it is recommended to use this checkpoint as the initial checkpoint,
or train with fixed frame rate first, then init from this checkpoint and move to flex frame rate in order to train stabily. 

2. The training launch script is:
- python train.py --config-name=codec_train

3. For multi-node training, use torchrun to launch the training script.

Note: 
- The original flexicodec training implementation is in `zero_shot_tts_training/realtime_communication/taste_v2/modeling_dualcodec.py`
- However, it was used in the old server and might only be used as a reference for you.
After I switched server, I lost access to the exact reproduction encironment (and had to use different data), and have not fully tested out this codebase for full reproduction. 
- I recently briefly tested the code, and the resulting model seems to have lower acoustic quality than anticipated, but semantic quality is good. Due to the limited time, I did not fully investigate the cause.
- I recommend tune the mel spectrogram reconstruction settings for better acoustic quality.
- One suggestion is: if you trained DAC and get satisfying result, then you can use the same settings for flexicodec (if you are training with the same data and sampling rate).

Tips for training flexicodec:
- When training flexicodec, it is recommended to train with fixed frame rate first, verify it's all right, then init from this checkpoint and move to flex frame rate in order to train stabily. (this is where https://huggingface.co/jiaqili3/flexicodec/blob/main/dualcodec_with_sensevoice_12hz_soundstream.safetensors is used for)
- For FlexiCodec training, the idea is to switch DualCodec with another semantic encoder “SenseVoice” and train it, and then add flexible-frame-rate modules and continue training. So the two projects are related.
- The training of DualCodec is built on DAC. If you have trained a DAC (or DualCodec, or other codec) before and already get good performance, then its hyperparameters/loss functions (like loss settings) can be directly migrated to these two codecs training.

Contributing to the codebase:

- If you find any issues or have any suggestions, please feel free to open an issue or a pull request.
- If you have any questions, please feel free to contact me.
- If you want to contribute to the training, please feel free to open an issue or a pull request.