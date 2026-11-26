#!/usr/bin/env python3
"""
NMC accelerator web console. Zero dependencies (Python 3.6+ standard library).

Run ON THE BOARD (needs the same rights you use in the Jupyter terminal):
    sudo python3 server.py
Then open  http://<board-ip>:8080  from any browser on the same network.
"""
import json, os, re, shlex, signal, socket, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
UP = ROOT / "uploads"
UP.mkdir(exist_ok=True)
MAX_UPLOAD = 25 * 1024 * 1024

lock = threading.Lock()
S = dict(status="idle", stage="idle", phase=None, log=[], started=None, ended=None,
         error=None, image=None, result=None, bringup_done=False, skipped_bringup=False,
         run_id=0, demo=False)


def cfg():
    return json.loads((ROOT / "config.json").read_text())


def log(line):
    with lock:
        S["log"].append(line)


def mark(line, c):
    for m in c.get("stage_markers", []):
        if re.search(m["regex"], line, re.I):
            with lock:
                S["phase"] = m["phase"]


# ---------------------------------------------------------------- running
def run_shell(cmd, c, timeout):
    log("$ " + cmd)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(["bash", "-c", cmd], cwd=c["workdir"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
                         errors="replace", env=env, start_new_session=True)
    timer = threading.Timer(timeout, lambda: os.killpg(p.pid, signal.SIGKILL))
    timer.start()
    out = []
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            out.append(line)
            log(line)
            mark(line, c)
        p.wait()
    finally:
        timer.cancel()
    if p.returncode == -9:
        log("[console] timed out and was stopped")
    return p.returncode, out


DEMO = [
    "Programming bitstream nmc_top.bit ... done",
    "NMC cluster ready: 4 banks, 16 PEs per unit",
    "Loading IFMap tiles into bank SRAMs",
    "Loading weights",
    "Running convolution layers on NMC units",
    "conv1_1 done", "conv1_2 done", "conv2_1 done", "conv2_2 done",
    "Storing OFMap and draining results",
    "Total cycles: 1843920", "Clock: 200 MHz", "Latency: 9.22 ms",
    "PE activity: 94.1 %", "Throughput: 96.0 GOPS", "Energy efficiency: 1828 GOPS/W",
    "Stalls: 1203", "Blocked cycles: 311",
    "PERF crit=1843920 busy_sum=7211034 mac=1740112 ops=26004480 stall=1203 blocked=311",
    "1. tabby cat: 61.4%", "2. tiger cat: 22.9%", "3. Egyptian cat: 9.8%",
    "4. lynx: 3.1%", "5. Persian cat: 1.2%",
]


def parse(lines, c):
    KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-/%\.\(\)]{1,48}?)\s*[:=]\s*(\S.*?)\s*$")
    PAIR = re.compile(r"\b([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?)\b")
    PROMO = re.compile(r"gops|latency|cycles|efficien|activity|fps|throughput|time|power|energy|stall", re.I)
    text = "\n".join(lines)
    details, preds, hl = {}, [], []
    top = re.compile(c.get("prediction_regex", "$^"), re.M)

    for ln in lines:
        if top.match(ln) and len(preds) < 5:
            m = top.match(ln)
            try:
                preds.append({"label": m.group(1).strip(), "score": float(m.group(2))})
                continue
            except ValueError:
                pass
        pairs = PAIR.findall(ln)
        m = KV.match(ln)
        if m and len(pairs) < 2 and "http" not in m.group(1).lower() \
                and not re.search(r"\d", m.group(1)) and len(m.group(2)) <= 60:
            details[m.group(1).strip()] = m.group(2)
        for k, v in pairs:
            details[k] = v
    if preds and all(p["score"] <= 1.0 for p in preds):
        for p in preds:
            p["score"] *= 100

    seen = set()
    for h in c.get("highlights", []):
        m = re.search(h["regex"], text, re.I)
        if m:
            hl.append({"label": h["label"], "value": m.group(1), "unit": h.get("unit", "")})
            seen.add(h["label"].lower())
    for k, v in details.items():
        if len(hl) >= 8:
            break
        stem = k.lower()[:5]
        if PROMO.search(k) and len(v) <= 24 and not any(x[:5] == stem or k.lower() in x or x in k.lower() for x in seen):
            hl.append({"label": k, "value": v, "unit": ""})
            seen.add(k.lower())

    for h in hl:
        details.setdefault(h["label"], h["value"] + (" " + h["unit"] if h["unit"] else ""))
    layers, header = [], None
    LAY = re.compile(r"^\s*((?:conv|fc|pool)\w*)\s+(\S+)\s+(.+?)\s*$", re.I)
    for ln in lines:
        if header is None and re.match(r"^\s*layer\b", ln, re.I):
            header = ln.split()
        m = LAY.match(ln)
        if m:
            layers.append({"name": m.group(1), "shape": m.group(2), "cols": m.group(3).split()})
    vm = re.search(c.get("verdict_regex", "$^"), text, re.I | re.M)

    rj = c.get("result_json")
    if rj:
        p = Path(rj) if os.path.isabs(rj) else Path(c["workdir"]) / rj
        if p.exists():
            try:
                j = json.loads(p.read_text())
                preds = [{"label": x["label"], "score": float(x["score"])} for x in j.get("predictions", [])][:5] or preds
                for k, v in j.get("metrics", {}).items():
                    hl.append({"label": k, "value": str(v), "unit": ""})
                details.update({k: str(v) for k, v in j.get("details", {}).items()})
            except Exception as e:
                log("[console] could not read result_json: %s" % e)
    return {"predictions": preds, "highlights": hl, "details": details, "layers": layers,
            "layer_header": header, "verdict": vm.group(1) if vm else None}


def job(image, force_bringup):
    c = cfg()
    try:
        S["demo"] = bool(c.get("demo"))
        need = c.get("bringup_cmd") and (force_bringup or not (c.get("bringup_once", True) and S["bringup_done"]))
        if not c.get("demo"):
            for key in ("bringup_cmd", "run_cmd"):
                if "EDIT_ME" in c.get(key, ""):
                    raise RuntimeError("config.json still has a placeholder for '%s'" % key)
        if need:
            S["stage"] = "bringup"
            if c.get("demo"):
                for l in DEMO[:2]:
                    log(l); time.sleep(1.2)
            else:
                rc, _ = run_shell(c["bringup_cmd"], c, c.get("bringup_timeout", 300))
                if rc:
                    raise RuntimeError("Bring-up failed (exit code %s). See the console." % rc)
            S["bringup_done"] = True
        else:
            S["skipped_bringup"] = True
        if image:
            S["stage"] = "run"
            rj = c.get("result_json")
            if rj:
                p = Path(rj) if os.path.isabs(rj) else Path(c["workdir"]) / rj
                try:
                    p.unlink()
                except OSError:
                    pass
            t0 = time.time()
            if c.get("demo"):
                out = []
                for l in DEMO[2:]:
                    log(l); out.append(l); mark(l, c); time.sleep(0.45)
            else:
                cmd = c["run_cmd"].replace("{image}", shlex.quote(str(image)))
                rc, out = run_shell(cmd, c, c.get("run_timeout", 900))
                if rc:
                    raise RuntimeError("Run failed (exit code %s). See the console." % rc)
            S["stage"] = "parse"
            res = parse(out, c)
            res["wall_seconds"] = round(time.time() - t0, 2)
            res["demo"] = bool(c.get("demo"))
            S["result"] = res
        S["status"], S["stage"] = "done", "done"
    except Exception as e:
        S["status"], S["error"] = "error", str(e)
        log("[console] " + str(e))
    finally:
        S["ended"] = time.time()


def start(image, force_bringup=False):
    with lock:
        if S["status"] == "running":
            return False
        S.update(status="running", stage="queued", phase=None, log=[], started=time.time(), ended=None,
                 error=None, image=os.path.basename(image) if image else None, result=None,
                 skipped_bringup=False, run_id=S["run_id"] + 1)
    threading.Thread(target=job, args=(image, force_bringup), daemon=True).start()
    return True


# ---------------------------------------------------------------- http
def sysinfo():
    ip = "?"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]; s.close()
    except Exception:
        pass
    temp = None
    try:
        temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        pass
    return {"host": socket.gethostname(), "ip": ip, "temp": temp}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self.send(200, (ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/status":
            q = parse_qs(u.query)
            since = int(q.get("since", ["0"])[0])
            if int(q.get("run", ["-1"])[0]) != S["run_id"]:
                since = 0
            with lock:
                end = S["ended"] or time.time()
                d = {k: S[k] for k in ("status", "stage", "phase", "error", "image", "result",
                                       "bringup_done", "skipped_bringup", "run_id", "demo")}
                d["log"] = S["log"][since:]
                d["next"] = len(S["log"])
                d["elapsed"] = round(end - S["started"], 1) if S["started"] else 0
            d["title"] = cfg().get("title", "NMC Accelerator")
            d["system"] = sysinfo()
            return self.send(200, d)
        self.send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if self.path == "/api/run":
            if n <= 0 or n > MAX_UPLOAD:
                return self.send(400, {"error": "Image missing or larger than 25 MB."})
            data = self.rfile.read(n)
            if data[:3] == b"\xff\xd8\xff":
                ext = ".jpg"
            elif data[:8] == b"\x89PNG\r\n\x1a\n":
                ext = ".png"
            else:
                return self.send(400, {"error": "Please upload a JPG or PNG image."})
            path = UP / ("input_%d%s" % (int(time.time()), ext))
            path.write_bytes(data)
            if not start(str(path)):
                return self.send(409, {"error": "A run is already in progress."})
            return self.send(200, {"ok": True})
        if self.path == "/api/bringup":
            if not start(None, True):
                return self.send(409, {"error": "A run is already in progress."})
            return self.send(200, {"ok": True})
        self.send(404, {"error": "not found"})


if __name__ == "__main__":
    port = cfg().get("port", 8080)
    print("NMC console on http://%s:%d   (Ctrl+C to stop)" % (sysinfo()["ip"], port))
    if cfg().get("demo"):
        print("DEMO MODE: no board commands are run. Set \"demo\": false in config.json.")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
