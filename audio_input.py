# audio_input.py
import threading, time, numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

def _hann(n): 
    return 0.5 - 0.5*np.cos(2*np.pi*np.arange(n)/max(1, n-1))

class AudioMeter:
    """
    Robust system-audio meter using sounddevice (WASAPI loopback).
      • Probes every WASAPI output; keeps the first that yields signal.
      • Falls back to mic only if all loopback devices fail.
    Public fields:
      .level   float [0..1]
      .bands   np.ndarray(8,) [0..1]
      .source  'loopback' | 'mic' | 'none'
      .device  str
      ._ok     bool
    """
    def __init__(self, samplerate=48000, blocksize=1024, channels=2, preferred_output=None, verbose=True):
        self.level  = 0.0
        self.bands  = np.zeros(8, float)
        self.source = "none"
        self.device = "n/a"
        self._ok    = False

        self._sr   = int(samplerate)
        self._bs   = int(blocksize)
        self._ch   = int(channels)
        self._pref = (preferred_output or "").lower()
        self._verbose = bool(verbose)

        self._attack  = 0.25
        self._release = 0.05

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thr  = None

        if sd is None:
            print("[AudioMeter] sounddevice not installed; audio reactive disabled.")
            return

        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def close(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.5)

    # ---------- internals ----------
    def _wasapi_id(self):
        try:
            for i, h in enumerate(sd.query_hostapis()):
                if "wasapi" in h["name"].lower():
                    return i
        except Exception:
            pass
        return None

    def _list_wasapi_outputs(self):
        wid = self._wasapi_id()
        if wid is None: return []
        outs=[]
        try:
            for dev in sd.query_devices():
                if dev["hostapi"] == wid and dev.get("max_output_channels",0) > 0:
                    outs.append(dev)
        except Exception:
            pass
        return outs

    def _sorted_candidates(self):
        """
        Prefer devices matching preferred_output substring, then default output, then others.
        """
        cands = self._list_wasapi_outputs()
        if not cands: return []

        deflt_idx = None
        try:
            deflt_idx = sd.default.device[1]  # (input, output)
        except Exception:
            pass

        ordered=[]
        for d in cands:
            name = d.get("name","?")
            score = 0
            if self._pref and self._pref in name.lower(): score += 100
            try:
                if deflt_idx is not None and sd.query_devices(deflt_idx)["name"] == name:
                    score += 50
            except Exception:
                pass
            ordered.append(( -score, name.lower(), d))
        ordered.sort()
        return [d for _,__,d in ordered]

    @staticmethod
    def _smooth(prev, target, k_up, k_down, dt):
        prev_arr = np.asarray(prev, float)
        targ_arr = np.asarray(target, float)
        rising = float(np.mean(targ_arr) - np.mean(prev_arr)) > 0.0
        k = float(k_up) if rising else float(k_down)
        alpha = 1.0 - np.exp(-k * float(dt))
        return prev_arr + alpha * (targ_arr - prev_arr)

    def _run(self):
        self._ok = True  # thread running (even if we end up with no device)
        win  = _hann(self._bs)
        nfft = self._bs
        band_edges = np.geomspace(60, 12000, 9)  # 8 bands
        eps = 1e-8
        state = {"lvl":0.0, "bands":np.zeros(8,float), "t":time.time()}

        def process_block(x, sr):
            x = x.mean(axis=1) * win[:len(x)]
            lvl = float(np.sqrt((x*x).mean() + eps))
            spec  = np.abs(np.fft.rfft(x, n=nfft))
            freqs = np.fft.rfftfreq(nfft, 1.0/sr)
            bands = np.zeros(8, float)
            for i in range(8):
                lo, hi = band_edges[i], band_edges[i+1]
                idx = (freqs >= lo) & (freqs < hi)
                bands[i] = 0.0 if not np.any(idx) else float(np.sqrt(np.mean(spec[idx]**2) + eps))
            # normalize
            bmax = np.percentile(bands, 95) + 1e-6
            bands = np.clip(bands / bmax, 0.0, 1.0)
            lvl = min(lvl * 8.0, 1.0)
            return lvl, bands

        def cb_factory(sr):
            def cb(indata, frames, time_info, status):
                if status:
                    # could print occasionally
                    pass
                lvl, bands = process_block(indata.copy(), sr)
                now = time.time()
                dt  = max(1e-3, now - state["t"]); state["t"] = now
                state["lvl"]   = self._smooth(state["lvl"],   lvl,   self._attack, self._release, dt)
                state["bands"] = self._smooth(state["bands"], bands, self._attack, self._release, dt)
                with self._lock:
                    self.level = float(state["lvl"])
                    self.bands = np.asarray(state["bands"], float).copy()
            return cb

        # --- Try WASAPI loopback on all outputs ---
        stream = None
        candidates = self._sorted_candidates()
        if self._verbose:
            labels = ", ".join(f"{d.get('name','?')}/{int(d.get('default_samplerate', self._sr))}Hz" for d in candidates)
            print(f"[AudioMeter] WASAPI candidates: {labels}")

        for dev in candidates:
            try:
                rate = int(dev.get("default_samplerate", self._sr))
                ch   = max(1, min(2, int(dev.get("max_output_channels", 2))))
                extra = sd.WasapiSettings(loopback=True)
                test_cb = cb_factory(rate)
                test_stream = sd.InputStream(
                    device=dev.get("name","?"),
                    samplerate=rate,
                    channels=ch,
                    blocksize=self._bs,
                    dtype="float32",
                    callback=test_cb,
                    extra_settings=extra,
                )
                test_stream.start()
                # Probe ~600ms for changing, non-zero level
                t0 = time.time(); last = None; ok = False
                while time.time() - t0 < 0.6 and not self._stop.is_set():
                    time.sleep(0.05)
                    with self._lock:
                        lvl = self.level
                    if last is None: last = lvl
                    if lvl > 1e-3 and abs(lvl - last) > 5e-4:
                        ok = True; break
                if ok:
                    stream = test_stream
                    self.source = "loopback"
                    self.device = dev.get("name","?")
                    if self._verbose:
                        print(f"[AudioMeter] Using LOOPBACK: {self.device} @ {rate}Hz")
                    break
                test_stream.stop(); test_stream.close()
            except Exception:
                try:
                    test_stream.stop(); test_stream.close()
                except Exception:
                    pass
                continue

        # --- Fallback: mic ---
        if stream is None:
            try:
                stream = sd.InputStream(
                    samplerate=self._sr,
                    channels=min(2, self._ch),
                    blocksize=self._bs,
                    dtype="float32",
                    callback=cb_factory(self._sr),
                )
                stream.start()
                self.source = "mic"
                self.device = "default input"
                if self._verbose:
                    print("[AudioMeter] Using MIC fallback (no loopback device produced signal).")
            except Exception as e:
                print(f"[AudioMeter] Failed to open any audio input: {e}")
                self._ok = False
                return

        # --- Pump until stop ---
        try:
            while not self._stop.is_set():
                time.sleep(0.02)
        finally:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
