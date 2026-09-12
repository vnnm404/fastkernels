"""Replay decoded media from a private audit cache across engines and versions."""
import functools
import hashlib
import json
import os
from pathlib import Path
import pickle


def _checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen_media(loader):
    @functools.wraps(loader)
    def load(*args, **kwargs):
        directory = os.environ.get('FASTKERNELS_MEDIA_CACHE')
        if not directory:
            return loader(*args, **kwargs)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = hashlib.sha256(json.dumps([loader.__name__, args, kwargs], sort_keys=True).encode()).hexdigest()
        path, stamp = root / (key + '.pickle'), root / (key + '.sha256')
        if path.exists() or stamp.exists():
            sha = _checksum(path)
            if sha != stamp.read_text().strip():
                raise ValueError('Frozen media checksum mismatch: ' + key)
            # Only read our own, private task cache; never user-supplied pickles.
            with path.open("rb") as stream:
                result = pickle.load(stream)
        else:
            result = loader(*args, **kwargs)
            if not result:
                raise ValueError('Media dataset produced no usable requests')
            temp = path.with_suffix('.tmp')
            with temp.open("wb") as stream:
                pickle.dump(result, stream, protocol=5)
            temp.replace(path)
            sha = _checksum(path)
            stamp.write_text(sha + '\n')
        print('Frozen media:', key, sha, flush=True)
        return result
    return load


def media_manifest():
    directory = os.environ.get('FASTKERNELS_MEDIA_CACHE')
    if not directory:
        return {}
    result = {}
    for stamp in sorted(Path(directory).glob('*.sha256')):
        sha = _checksum(stamp.with_suffix('.pickle'))
        if sha != stamp.read_text().strip():
            raise ValueError('Frozen media checksum mismatch')
        result[stamp.stem] = sha
    return result


def preload_media(throughput, latency, seed, loader, whisper=False):
    """Populate the same cache keys workers use, before allocating any engine."""
    for scenario in throughput:
        if 'dataset' in scenario and 'prompt_token_ids' not in scenario:
            loader(scenario['dataset'], scenario['dataset_split'], scenario['num_seqs'], seed)
    for scenario in latency:
        if 'dataset' in scenario and 'prompt_token_ids' not in scenario:
            count = scenario['batch_size'] if whisper else 1
            loader(scenario['dataset'], scenario['dataset_split'], count, seed + 200 if whisper else seed)
