import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import torchaudio
from torchaudio.functional import resample

from tqdm import tqdm
import subprocess


def process_wav(in_path, out_path, sample_rate):
    # wav, sr = torchaudio.load(in_path)
    # wav = resample(wav, sr, sample_rate)
    out_path = Path(str(out_path).replace('mp4', 'wav'))
    ffmpeg_process = subprocess.Popen(
        ['ffmpeg', '-y', '-i', in_path, '-ac', '1', '-ar', str(sample_rate), out_path],
        stdout=-1, stderr=-1, text=True
    )
    stdout, stderr = ffmpeg_process.communicate(None, timeout=5.0)
    ffmpeg_process.kill()
    # torchaudio.save(out_path, wav, sample_rate)
    return out_path, 0# wav.size(-1) / sample_rate


def preprocess_dataset(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    print(f"Resampling audio in {args.in_dir}")
    for in_path in args.in_dir.rglob("*.mp4"):
        relative_path = in_path.relative_to(args.in_dir)
        out_path = args.out_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # process_wav(in_path, out_path, args.sample_rate)
        futures.append(
            executor.submit(process_wav, in_path, out_path, args.sample_rate)
        )
        # import pdb; pdb.set_trace()

    results = [future.result() for future in tqdm(futures)]

    lengths = {path.stem: length for path, length in results}
    seconds = sum(lengths.values())
    hours = seconds / 3600
    print(f"Wrote {len(lengths)} utterances ({hours:.2f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample an audio dataset.")
    parser.add_argument(
        "in_dir", metavar="in-dir", help="path to the dataset directory.", type=Path
    )
    parser.add_argument(
        "out_dir", metavar="out-dir", help="path to the output directory.", type=Path
    )
    parser.add_argument(
        "--sample-rate",
        help="target sample rate (default 16kHz)",
        type=int,
        default=16000,
    )
    args = parser.parse_args()
    preprocess_dataset(args)
