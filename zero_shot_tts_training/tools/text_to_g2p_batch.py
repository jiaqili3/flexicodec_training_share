import os

# Replace with your actual G2P function
from g2p_phonemizer import g2p_phonemizer_en  

directory = "/gluster-ssd-tts/jiaqi_repos/seed_eval/0305_soundstorm25hz_noise_augment_g2p/seedtts_en_0.9_top10_top1.0/rerank_1"

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        with open(file_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                continue  # Skip empty files

            original_text = lines[0].strip()
            ipa_line = g2p_phonemizer_en(original_text)
            
            # Rewind to overwrite with original + IPA
            f.seek(0)
            f.write(original_text + "\n" + ipa_line + "\n")
            f.truncate()

print("Done adding IPA lines to all txt files.")
