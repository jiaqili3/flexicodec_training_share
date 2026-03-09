import json
import subprocess
import string
import re
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from g2p_en import G2p
import librosa
from scipy.interpolate import interp1d
import nltk
# nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

PHONE_DICT_PATH = "CSDs/data/block/voicebox/phones.txt"


def load_str_to_int_dict(dict_path):
    str_to_int_dict = {}
    with open(dict_path, "r") as file:
        for line in file:
            line = line.strip().split()
            str_to_int_dict[line[0]] = int(line[1])
    return str_to_int_dict


def text_2_phone_seq(
    text: str,
    tts_tool_dir: str,
):
    if tts_tool_dir == "":
        return text_2_kaldi_phone_seq(text)
    else:
        return text_2_tts_phone_seq(text, tts_tool_dir)


def text_2_kaldi_phone_seq(
    text: str,
):
    # load the phone dictionary
    str_to_int_dict = load_str_to_int_dict(PHONE_DICT_PATH)
    g2p = G2p()
    sil_prob = 0
    l = text
    words = l.strip().upper()
    words = re.sub("[" + string.punctuation.replace("'", "") + "]", " ", words).strip()  # wo_dash
    words = words.split()
    phones = []
    phones.extend(["SIL"])
    sample_sil_probs = None
    if sil_prob > 0 and len(words) > 1:
        sample_sil_probs = np.random.random(len(words) - 1)
    for i, w in enumerate(words):
        out = g2p(w)
        if w == "A":
            out = ["EY1"]
        out = [phn for phn in out if phn not in [" ", "'"]]
        if len(out) == 0:
            continue
        if len(out) == 1:
            out[0] = out[0] + "_S"
        else:
            out = [phn + "_I" for phn in out]
            out[0] = out[0][:-2] + "_B"
            out[-1] = out[-1][:-2] + "_E"
        phones.extend(out)
        if (
            sample_sil_probs is not None
            and i < len(sample_sil_probs)
            and sample_sil_probs[i] < sil_prob
        ):
            phones.extend(["SIL"])
    phones.extend(["SIL"])
    idxs = [str_to_int_dict[p] for p in phones]
    return phones, idxs


def text_2_tts_phone_seq(
    text: str,
    tts_tool_dir: str,
):
    with open(f"{tts_tool_dir}/phone_set.json", "r") as fp:
        phone2idx = json.load(fp)

    with NamedTemporaryFile() as tmp_text, NamedTemporaryFile() as tmp_xml:
        tmp_text.write(text.encode("utf-8"))
        tmp_text.flush()

        subprocess.call(
            f"{tts_tool_dir}/ttsdumptool -m dumpwordmetadata -i {tmp_text.name} -o {tmp_xml.name} -s {tts_tool_dir}/enus_voice/dump_phone_cpu.json",
            shell=True,
        )

        def traverse_xml_tree(node):
            result = []
            if node.tag == "{http://schemas.microsoft.com/tts}w":
                p = node.attrib["p"]
                if node.attrib["type"] == "normal":
                    result.append("br1")
                    result.extend([f"en-us_{_p}" for _p in p.split(" ") if _p != "-"])
                elif node.attrib["type"] == "punc":
                    result.extend([p])
            for child in node:
                result.extend(traverse_xml_tree(child))
            return result

        xml_tree = ET.parse(tmp_xml.name)
        phones = traverse_xml_tree(xml_tree.getroot())
        if phones[0] == "br1":
            phones = phones[1:]
        idxs = [phone2idx[p] for p in phones]

    return phones, idxs


def per_phone_duration_gen(z, ghost_sil_point=None):
    """
    Generate the per-phone duration sequence and the phone sequence based on the frame-level phone transcript
    args:
        z: the frame-level phone transcript, a list of phone ids(int)
        ghost_sil_point: the frame to insert ghost silence, True/False
    return:
        l: the per-phone duration sequence, a numpy array of int
        y: the phone sequence, a numpy array of int
        gs: if ghost silence should be inserted, an array of bool
    """
    l = []
    phones = []
    gs = [False]
    prev_phone = z[0]
    count = 0
    for i, phone in enumerate(z):
        if phone == prev_phone:
            count += 1
        else:
            l.append(count)
            phones.append(prev_phone)
            prev_phone = phone
            count = 1

            if ghost_sil_point is None:
                gs.append(False)
            else:
                if ghost_sil_point[i] == True and (i == 0 or ghost_sil_point[i - 1] == False):
                    gs.append(True)
                else:
                    gs.append(False)
    l.append(count)
    phones.append(prev_phone)

    # apply transformation to the duration sequence
    # refer to https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 22, section "A.3 Data transformation"
    l_new = []
    for x in l:
        x_low = x - 0.5
        x_high = x + 0.5
        # uniformly sample a number from [x_low, x_high)
        x_new = np.random.uniform(x_low, x_high)
        # tranform to log scale
        x_new = np.log(1 + x_new, dtype=np.float32)
        l_new.append(x_new)

    return np.array(l), np.array(l_new), np.array(phones), gs


def get_gain(
    mel: torch.Tensor,
):
    """
    get the gain for the mel spectrum
    args:
        mel: the mel spectrum, a tensor of float, [M, n_mel_channels]
    return:
        gain: the gain for the mel spectrum, a tensor of float
    """
    energy = torch.exp(6 * mel - 5) ** 2
    # exclude the near silent frames
    energy = energy[energy > 1.0e-4]
    gain = torch.mean(energy)
    return gain.item()


def normalize_mel(
    mel_in: torch.Tensor,
    norm_st: int,
    norm_ed: int,
):
    """
    normalize the mel spectrum between norm_st and norm_ed to make the mel spectrum smooth
    args:
        mel_in: the input mel spectrum, a tensor of float, [M, n_mel_channels]
        norm_st: the start frame id of the normalization, an int
        norm_ed: the end frame id of the normalization, an int
    return:
        mel_out: the normalized mel spectrum, a tensor of float, [M, n_mel_channels]
    """
    mel_out = mel_in.clone()
    if norm_st == 0:
        left_gain = 0
    else:
        left_gain = get_gain(mel_in[:norm_st])
    middle_gain = get_gain(mel_in[norm_st:norm_ed])
    if norm_ed == mel_in.shape[0]:
        right_gain = 0
    else:
        right_gain = get_gain(mel_in[norm_ed:])
    print(
        f"\033[91m energy difference (left: {left_gain:.4f}, middle: {middle_gain:.4f}, right: {right_gain:.4f}) \033[0m"
    )
    mel_out[norm_st:norm_ed] += (
        torch.log(torch.tensor(max(left_gain, right_gain) / middle_gain)) / 12.0
    )
    return mel_out


def dump_info(out_path: str, info=None):
    out_info_path = re.sub(r".wav$", ".info", out_path)
    assert out_path != out_info_path

    if len(info.keys()) > 0:
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].cpu().numpy().tolist()
            elif isinstance(info[key], np.ndarray):
                info[key] = info[key].tolist()

        with open(out_info_path, "w") as fp:
            fp.write(json.dumps(info))


def reformat_audio(audio, orig_sr, target_sr):
    assert len(audio.shape) < 3

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        audio = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best"
        )
    return audio


def frame_sync(z, num_frames, axis=0, astype="int32"):
    # NOTE:  This function needs to be aligned with audio_phone_sync()
    #        in scades/data/block/voicebox/audio_phone_sync.py that is used for training.
    f = interp1d(np.linspace(0, 1, len(z)), z, axis=0, kind="nearest")
    z_sync = f(np.linspace(0, 1, num_frames))

    return z_sync.astype(astype)
import json
import subprocess
import string
import re
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from g2p_en import G2p
import librosa
from scipy.interpolate import interp1d
import os
PHONE_DICT_PATH = f"{os.path.dirname(__file__)}/phone_seq.txt"


def load_str_to_int_dict(dict_path):
    str_to_int_dict = {}
    with open(dict_path, "r") as file:
        for line in file:
            line = line.strip().split()
            str_to_int_dict[line[0]] = int(line[1])
    return str_to_int_dict


def text_2_phone_seq(
    text: str,
    tts_tool_dir: str="",
):
    if tts_tool_dir == "":
        return text_2_kaldi_phone_seq(text)
    else:
        return text_2_tts_phone_seq(text, tts_tool_dir)


def text_2_kaldi_phone_seq(
    text: str,
):
    # load the phone dictionary
    str_to_int_dict = load_str_to_int_dict(PHONE_DICT_PATH)
    g2p = G2p()
    sil_prob = 0
    l = text
    words = l.strip().upper()
    words = re.sub("[" + string.punctuation.replace("'", "") + "]", " ", words).strip()  # wo_dash
    words = words.split()
    phones = []
    phones.extend(["SIL"])
    sample_sil_probs = None
    if sil_prob > 0 and len(words) > 1:
        sample_sil_probs = np.random.random(len(words) - 1)
    for i, w in enumerate(words):
        out = g2p(w)
        if w == "A":
            out = ["EY1"]
        out = [phn for phn in out if phn not in [" ", "'"]]
        if len(out) == 0:
            continue
        if len(out) == 1:
            out[0] = out[0] + "_S"
        else:
            out = [phn + "_I" for phn in out]
            out[0] = out[0][:-2] + "_B"
            out[-1] = out[-1][:-2] + "_E"
        phones.extend(out)
        if (
            sample_sil_probs is not None
            and i < len(sample_sil_probs)
            and sample_sil_probs[i] < sil_prob
        ):
            phones.extend(["SIL"])
    phones.extend(["SIL"])
    idxs = [str_to_int_dict[p] for p in phones]
    return phones, idxs


def text_2_tts_phone_seq(
    text: str,
    tts_tool_dir: str,
):
    with open(f"{tts_tool_dir}/phone_set.json", "r") as fp:
        phone2idx = json.load(fp)

    with NamedTemporaryFile() as tmp_text, NamedTemporaryFile() as tmp_xml:
        tmp_text.write(text.encode("utf-8"))
        tmp_text.flush()

        subprocess.call(
            f"{tts_tool_dir}/ttsdumptool -m dumpwordmetadata -i {tmp_text.name} -o {tmp_xml.name} -s {tts_tool_dir}/enus_voice/dump_phone_cpu.json",
            shell=True,
        )

        def traverse_xml_tree(node):
            result = []
            if node.tag == "{http://schemas.microsoft.com/tts}w":
                p = node.attrib["p"]
                if node.attrib["type"] == "normal":
                    result.append("br1")
                    result.extend([f"en-us_{_p}" for _p in p.split(" ") if _p != "-"])
                elif node.attrib["type"] == "punc":
                    result.extend([p])
            for child in node:
                result.extend(traverse_xml_tree(child))
            return result

        xml_tree = ET.parse(tmp_xml.name)
        phones = traverse_xml_tree(xml_tree.getroot())
        if phones[0] == "br1":
            phones = phones[1:]
        idxs = [phone2idx[p] for p in phones]

    return phones, idxs


def per_phone_duration_gen(z, ghost_sil_point=None):
    """
    Generate the per-phone duration sequence and the phone sequence based on the frame-level phone transcript
    args:
        z: the frame-level phone transcript, a list of phone ids(int)
        ghost_sil_point: the frame to insert ghost silence, True/False
    return:
        l: the per-phone duration sequence, a numpy array of int
        y: the phone sequence, a numpy array of int
        gs: if ghost silence should be inserted, an array of bool
    """
    l = []
    phones = []
    gs = [False]
    prev_phone = z[0]
    count = 0
    for i, phone in enumerate(z):
        if phone == prev_phone:
            count += 1
        else:
            l.append(count)
            phones.append(prev_phone)
            prev_phone = phone
            count = 1

            if ghost_sil_point is None:
                gs.append(False)
            else:
                if ghost_sil_point[i] == True and (i == 0 or ghost_sil_point[i - 1] == False):
                    gs.append(True)
                else:
                    gs.append(False)
    l.append(count)
    phones.append(prev_phone)

    # apply transformation to the duration sequence
    # refer to https://dl.fbaipublicfiles.com/voicebox/paper.pdf, page 22, section "A.3 Data transformation"
    l_new = []
    for x in l:
        x_low = x - 0.5
        x_high = x + 0.5
        # uniformly sample a number from [x_low, x_high)
        x_new = np.random.uniform(x_low, x_high)
        # tranform to log scale
        x_new = np.log(1 + x_new, dtype=np.float32)
        l_new.append(x_new)

    return np.array(l), np.array(l_new), np.array(phones), gs


def get_gain(
    mel: torch.Tensor,
):
    """
    get the gain for the mel spectrum
    args:
        mel: the mel spectrum, a tensor of float, [M, n_mel_channels]
    return:
        gain: the gain for the mel spectrum, a tensor of float
    """
    energy = torch.exp(6 * mel - 5) ** 2
    # exclude the near silent frames
    energy = energy[energy > 1.0e-4]
    gain = torch.mean(energy)
    return gain.item()


def normalize_mel(
    mel_in: torch.Tensor,
    norm_st: int,
    norm_ed: int,
):
    """
    normalize the mel spectrum between norm_st and norm_ed to make the mel spectrum smooth
    args:
        mel_in: the input mel spectrum, a tensor of float, [M, n_mel_channels]
        norm_st: the start frame id of the normalization, an int
        norm_ed: the end frame id of the normalization, an int
    return:
        mel_out: the normalized mel spectrum, a tensor of float, [M, n_mel_channels]
    """
    mel_out = mel_in.clone()
    if norm_st == 0:
        left_gain = 0
    else:
        left_gain = get_gain(mel_in[:norm_st])
    middle_gain = get_gain(mel_in[norm_st:norm_ed])
    if norm_ed == mel_in.shape[0]:
        right_gain = 0
    else:
        right_gain = get_gain(mel_in[norm_ed:])
    print(
        f"\033[91m energy difference (left: {left_gain:.4f}, middle: {middle_gain:.4f}, right: {right_gain:.4f}) \033[0m"
    )
    mel_out[norm_st:norm_ed] += (
        torch.log(torch.tensor(max(left_gain, right_gain) / middle_gain)) / 12.0
    )
    return mel_out


def dump_info(out_path: str, info=None):
    out_info_path = re.sub(r".wav$", ".info", out_path)
    assert out_path != out_info_path

    if len(info.keys()) > 0:
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].cpu().numpy().tolist()
            elif isinstance(info[key], np.ndarray):
                info[key] = info[key].tolist()

        with open(out_info_path, "w") as fp:
            fp.write(json.dumps(info))


def reformat_audio(audio, orig_sr, target_sr):
    assert len(audio.shape) < 3

    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        audio = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best"
        )
    return audio


def frame_sync(z, num_frames, axis=0, astype="int32"):
    # NOTE:  This function needs to be aligned with audio_phone_sync()
    #        in scades/data/block/voicebox/audio_phone_sync.py that is used for training.
    f = interp1d(np.linspace(0, 1, len(z)), z, axis=0, kind="nearest")
    z_sync = f(np.linspace(0, 1, num_frames))

    return z_sync.astype(astype)

if __name__ == "__main__":
    phones, idxs = text_2_phone_seq("Hello, world!!  to be or not to be")
    print(phones)
    print(idxs)