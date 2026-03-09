import argparse
import torch
import torchaudio
from tqdm import tqdm
import onnxruntime
import torchaudio.compliance.kaldi as kaldi


def main(args):
    """
    :param args:
    :return:
    """
    utt2wav, utt2spk = {}, {}
    with open("{}/wav.scp".format(args.dir)) as f:
        for l in f:
            l = l.replace("\n", "").split()
            utt2wav[l[0]] = l[1]
    with open("{}/utt2spk".format(args.dir)) as f:
        for l in f:
            l = l.replace("\n", "").split()
            utt2spk[l[0]] = l[1]

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(
        args.onnx_path, sess_options=option, providers=providers
    )

    utt2embedding, spk2embedding = {}, {}
    for utt in tqdm(utt2wav.keys()):
        audio, sample_rate = torchaudio.load(utt2wav[utt])
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )(audio)
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            ort_session.run(
                None,
                {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
            )[0]
            .flatten()
            .tolist()
        )
        utt2embedding[utt] = embedding
        spk = utt2spk[utt]
        if spk not in spk2embedding:
            spk2embedding[spk] = []
        spk2embedding[spk].append(embedding)
    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    torch.save(utt2embedding, "{}/utt2embedding.pt".format(args.dir))
    torch.save(spk2embedding, "{}/spk2embedding.pt".format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--onnx_path", type=str)
    args = parser.parse_args()
    main(args)
