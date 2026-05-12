#!/usr/bin/env python3
"""Uloží 5 různých hlasů z YodaLingua-preview jako voice/ref_v{1..5}.wav -- user vybere ženský."""
import io
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from pydub import AudioSegment

HERE = Path(__file__).parent
PARQ = Path("/tmp/yoda-prev/data/Czech-00000-of-00001.parquet")

df = pq.read_table(PARQ).to_pandas()
df = df.sort_values("dnsmos", ascending=False)

seen = set()
idx = 0
for _, row in df.iterrows():
    sid = row["speaker_id"]
    if sid in seen:
        continue
    seen.add(sid)
    idx += 1
    mp3_bytes = row["mp3"]["bytes"]
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    arr = audio.get_array_of_samples()
    import numpy as np
    arr = np.array(arr, dtype=np.float32) / 32768.0
    if audio.channels == 2:
        arr = arr.reshape(-1, 2).mean(axis=1)
    sr = audio.frame_rate
    dur = len(arr) / sr
    out = HERE / f"ref_v{idx}.wav"
    sf.write(out, arr, sr)
    print(f"v{idx}: {out.name} ({dur:.1f}s) spk={sid} dnsmos={row['dnsmos']:.2f} text='{row['text'][:60]}'")
    if idx >= 5:
        break
