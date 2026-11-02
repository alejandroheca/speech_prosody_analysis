"""
Step 01: split every recording into speech and crowd.

    conda activate crowd_audio
    python 01_separate.py

Input:  config.RAW_AUDIO_DIR/<name>.wav|.flac, for every recording in
        config.AUDIO_METADATA
Output: config.DENOISED_DIR/denoised_<name>.flac   speech (GTCRN denoiser)
        config.CROWD_DIR/noise_<name>.flac         crowd = original - speech
        both mono, 16 kHz. Recordings with both outputs are skipped.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from prosodic_features import config
from prosodic_features.audio import find_audio_file
from prosodic_features.metadata import load_recordings
from prosodic_features.separation import init_worker, separate_in_worker


def main() -> None:
    if not config.GTCRN_MODEL_PATH.exists():
        raise FileNotFoundError(f"GTCRN model not found: {config.GTCRN_MODEL_PATH} "
                                "(see models/README.md)")
    config.DENOISED_DIR.mkdir(parents=True, exist_ok=True)
    config.CROWD_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for rec in load_recordings():
        audio_path = find_audio_file(config.RAW_AUDIO_DIR, "", rec.stem)
        if not audio_path.exists():
            print(f"  [missing] no audio for {rec.stem} in {config.RAW_AUDIO_DIR}")
            continue
        jobs.append((audio_path,
                     config.DENOISED_DIR / f"{config.DENOISED_PREFIX}{rec.stem}.flac",
                     config.CROWD_DIR / f"{config.CROWD_PREFIX}{rec.stem}.flac"))
    print(f"[Config] Input:   {config.RAW_AUDIO_DIR}")
    print(f"[Config] Speech:  {config.DENOISED_DIR}")
    print(f"[Config] Crowd:   {config.CROWD_DIR}")
    print(f"[Config] Workers: {config.SEPARATE_N_WORKERS} | {len(jobs)} recordings\n")

    t_start, total_audio = time.time(), 0.0
    with ProcessPoolExecutor(max_workers=config.SEPARATE_N_WORKERS, initializer=init_worker,
                             initargs=(config.GTCRN_MODEL_PATH,)) as executor:
        futures = {executor.submit(separate_in_worker, job): job[0] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            name, result = futures[future].name, future.result()
            if isinstance(result, str):
                print(f"[{i}/{len(jobs)}] {name} -> ERROR: {result}")
                continue
            skipped, duration, proc_time = result
            if skipped:
                print(f"[{i}/{len(jobs)}] {name} -> SKIP")
                continue
            total_audio += duration
            speed = duration / proc_time if proc_time > 0 else 0
            print(f"[{i}/{len(jobs)}] {name} | {duration:.2f}s audio | "
                  f"{proc_time:.2f}s proc | {speed:.2f}x realtime")

    print(f"\n[Done] {total_audio / 60:.2f} min of audio in "
          f"{(time.time() - t_start) / 60:.2f} min")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
