#!/usr/bin/env python3
"""Darvin training runner.

This script orchestrates the training lifecycle inside the container:
1) Read required DARVIN_* environment variables injected by backend
2) Send status callbacks to DARVIN_CALLBACK_URL
3) Download training bundle via DARVIN_TRAIN_BUNDLE_URL using job_id + job_token
4) Map output directory to DARVIN_OUTPUT_DIR and run existing training steps
5) Produce artifacts and manifest, then report completed or failed

Minimal dependencies: only standard library (urllib, subprocess, zipfile, json).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
import fnmatch
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlparse
import http.client
import zipfile


def _now_ts() -> int:
    return int(time.time())


def _env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"missing env: {name}")
    return v


def _ensure_https(u: Optional[str]) -> Optional[str]:
    if not u:
        return u
    try:
        if u.startswith("http://"):
            return "https://" + u[len("http://") :]
    except Exception:
        pass
    return u


def _post_json(url: str, obj: Dict[str, Any], timeout: int = 15, max_retries: Optional[int] = None) -> Dict[str, Any]:
    """POST JSON with http->https upgrade, redirect handling, and retries.

    Returns parsed JSON on success; raises on final failure.
    """
    url0 = url
    url = _ensure_https(url) or url
    data = json.dumps(obj).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "darvin-runner/1.0"}

    def _do_post(u: str) -> Dict[str, Any]:
        req = Request(u, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=timeout) as resp:  # nosec - controlled URL from backend
            raw = resp.read()
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                return {}

    # retry settings
    if max_retries is None:
        try:
            max_retries = int(os.getenv("DARVIN_RETRY_MAX", "5"))
        except Exception:
            max_retries = 5
    try:
        base_delay = float(os.getenv("DARVIN_RETRY_BASE_DELAY", "0.5"))
    except Exception:
        base_delay = 0.5

    attempt = 0
    last_exc: Optional[Exception] = None
    while attempt <= max_retries:
        try:
            if attempt > 0:
                print(f"[darvin-runner] retrying post_json attempt={attempt} url={url}", file=sys.stderr)
            return _do_post(url)
        except HTTPError as e:
            code = getattr(e, 'code', None)
            loc = None
            try:
                loc = e.headers.get("Location") if hasattr(e, "headers") and e.headers else None
            except Exception:
                loc = None
            retry_url = _ensure_https(loc) or _ensure_https(url0) or url
            print(f"[darvin-runner] post_json failed: code={code} url={url0} retry_to={retry_url}", file=sys.stderr)
            if loc and retry_url != url:
                url = retry_url
            if code not in (429, 500, 502, 503, 504):
                raise
            last_exc = e
        except URLError as e:
            print(f"[darvin-runner] post_json urlerror: url={url0} err={e}", file=sys.stderr)
            last_exc = e

        attempt += 1
        if attempt > max_retries:
            break
        sleep_s = min(base_delay * (2 ** (attempt - 1)), 8.0) + random.uniform(0, 0.25)
        time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("post_json_failed")


def _download_bundle(url: str, job_id: str, job_token: str, out_path: Path, timeout: int = 60) -> None:
    payload = {"job_id": job_id, "job_token": job_token}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "darvin-runner/1.0"}

    def _do_download(u: str) -> None:
        req = Request(u, data=data, headers=headers, method="POST")
        with urlopen(req, timeout=timeout) as resp:  # nosec - controlled URL from backend
            CHUNK = 1024 * 256
            with out_path.open("wb") as f:
                while True:
                    buf = resp.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)

    url0 = url
    url = _ensure_https(url) or url
    print(f"[darvin-runner] downloading bundle: url={url}", file=sys.stderr)
    try:
        _do_download(url)
        return
    except HTTPError as e:
        loc = None
        try:
            loc = e.headers.get("Location") if hasattr(e, "headers") and e.headers else None
        except Exception:
            loc = None
        retry_url = _ensure_https(loc) or _ensure_https(url0) or url
        print(f"[darvin-runner] retry bundle download due to HTTP {getattr(e, 'code', None)} -> {retry_url}", file=sys.stderr)
        # small retry loop for transient issues
        for attempt in range(1, 4):
            try:
                _do_download(retry_url)
                return
            except Exception as e2:
                print(f"[darvin-runner] bundle retry={attempt} to={retry_url} err={e2}", file=sys.stderr)
                time.sleep(min(2 ** attempt * 0.5, 4.0) + random.uniform(0, 0.2))
        raise
    except URLError as e:
        print(f"[darvin-runner] bundle urlerror: url={url0} err={e}", file=sys.stderr)
        raise


def _zip_has_training_datasets(z: Path) -> bool:
    try:
        with zipfile.ZipFile(z, "r") as zf:
            names = zf.namelist()
            return any(n.startswith("dataset/training/") and n.lower().endswith(".zip") for n in names)
    except Exception:
        return False


def _zip_list(z: Path) -> int:
    try:
        with zipfile.ZipFile(z, "r") as zf:
            return len(zf.namelist())
    except Exception:
        return 0


@dataclass
class Ctx:
    job_id: str
    job_token: str
    callback_url: str
    bundle_url: Optional[str]
    upload_url: Optional[str]
    output_dir: Path
    dataset_cids: Optional[str]
    hyperparams: Optional[str]
    subnet_address: Optional[str]
    work_dir: Path


class Runner:
    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self._stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self._progress_val = 0
        self._started_ts = _now_ts()
        self._ended_ts: Optional[int] = None

    # --------------------------- Callback helpers ---------------------------
    def _callback(self, status: str, progress: Optional[int] = None, message: Optional[str] = None, metrics: Optional[dict] = None, trained_model_cid: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "job_id": self.ctx.job_id,
            "job_token": self.ctx.job_token,
            "status": status,
        }
        if isinstance(progress, int):
            payload["progress"] = max(0, min(100, progress))
        if message:
            payload["message"] = message
        if metrics is not None:
            payload["metrics"] = metrics
        if trained_model_cid:
            payload["trained_model_cid"] = trained_model_cid
        try:
            print(f"[darvin-runner] callback status={status} progress={payload.get('progress')} msg_len={len(payload.get('message',''))} has_cid={bool(trained_model_cid)}", file=sys.stderr)
            _post_json(self.ctx.callback_url, payload, timeout=10)
        except Exception:
            # best-effort; do not raise
            pass

    def _hb_loop(self) -> None:
        # send training heartbeats with coarse progress up to 90
        while not self._stop.wait(30):
            self._progress_val = min(90, self._progress_val + 3)
            self._callback("training", progress=self._progress_val)

    def _hb_start(self) -> None:
        self._hb_thread = threading.Thread(target=self._hb_loop, name="darvin-train-hb", daemon=True)
        self._hb_thread.start()

    def _hb_stop(self) -> None:
        self._stop.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=5)

    # ----------------------------- Main steps -------------------------------
    def download_bundle(self) -> Optional[Path]:
        if not self.ctx.bundle_url:
            return None
        tmp = self.ctx.work_dir / f"bundle_{self.ctx.job_id}.zip"
        self._callback("downloading_bundle", progress=1)
        try:
            _download_bundle(self.ctx.bundle_url, self.ctx.job_id, self.ctx.job_token, tmp, timeout=300)
        except (HTTPError, URLError) as exc:
            self._callback("failed", message=f"download_http_error: {exc}")
            raise
        except Exception as exc:
            self._callback("failed", message=f"download_error: {exc}")
            raise
        if not tmp.exists() or _zip_list(tmp) == 0 or not _zip_has_training_datasets(tmp):
            self._callback("failed", message="invalid_training_bundle")
            raise RuntimeError("invalid_training_bundle")
        return tmp

    def run_training(self, bundle_path: Optional[Path]) -> None:
        # Map output dir
        os.makedirs(self.ctx.output_dir, exist_ok=True)
        os.environ["TARGET_MODEL_DIR"] = str(self.ctx.output_dir)
        # Provide dataset source to prepare script
        if bundle_path:
            os.environ["DARVIN_TRAIN_BUNDLE_PATH"] = str(bundle_path)
        # optional cids/hyperparams for manifest/logging
        if self.ctx.dataset_cids:
            os.environ["DARVIN_DATASET_CIDS"] = self.ctx.dataset_cids
        if self.ctx.hyperparams:
            os.environ["DARVIN_HYPERPARAMS"] = self.ctx.hyperparams

        # Start heartbeats
        self._callback("training", progress=5)
        self._hb_start()

        # Step 1: prepare config and dataset
        prep = subprocess.run(
            [sys.executable, "-u", "/tmp/workspace/train/train-0/prepare_v2.py"],
            cwd="/tmp/workspace/train",
            env=os.environ.copy(),
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        if prep.returncode != 0:
            self._hb_stop()
            self._callback("failed", message=f"prepare_failed: rc={prep.returncode}")
            raise RuntimeError("prepare_failed")

        # Step 2: run training
        train = subprocess.run(
            ["bash", "-lc", "cd /tmp/workspace/train && llamafactory-cli train train.yaml"],
            env=os.environ.copy(),
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        self._hb_stop()
        if train.returncode != 0:
            self._callback("failed", message=f"training_failed: rc={train.returncode}")
            raise RuntimeError("training_failed")

    def produce_artifacts(self) -> None:
        self._callback("uploading", progress=95)
        # manifest
        manifest = {
            "job_id": self.ctx.job_id,
            "dataset_cids": (self.ctx.dataset_cids or "").split(",") if self.ctx.dataset_cids else [],
            "hyperparams": self._safe_json(self.ctx.hyperparams),
            "subnet_address": self.ctx.subnet_address,
            "started_at": self._started_ts,
            "finished_at": _now_ts(),
            "output_dir": str(self.ctx.output_dir),
        }
        try:
            (self.ctx.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        # zip main outputs for convenience（固定排除 checkpoint-* 目录）
        zip_path = self.ctx.output_dir / "trained_model.zip"
        # 固定排除规则：路径的任一组件匹配 checkpoint-* 即跳过
        exclude_patterns = ["checkpoint-*"]

        def _excluded(rel: Path) -> bool:
            s = str(rel)
            for pat in exclude_patterns:
                if fnmatch.fnmatchcase(s, pat):
                    return True
                for part in rel.parts:
                    if fnmatch.fnmatchcase(part, pat):
                        return True
            return False
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in self.ctx.output_dir.rglob("*"):
                    if p.is_file() and p.name != zip_path.name:
                        rel = p.relative_to(self.ctx.output_dir)
                        if _excluded(rel):
                            # Skip excluded files quietly
                            continue
                        try:
                            zf.write(p, rel)
                        except Exception:
                            continue
        except Exception:
            # non-blocking
            pass

        # If upload URL is provided by backend, upload the artifact and carry back the CID
        tm_cid: Optional[str] = None
        if self.ctx.upload_url and zip_path.exists():
            try:
                print(f"[darvin-runner] start upload: path={zip_path} size_mb={zip_path.stat().st_size/1024/1024:.2f} url={self.ctx.upload_url}", file=sys.stderr)
                tm_cid = self._upload_trained_model(zip_path)
                print(f"[darvin-runner] upload done: cid={tm_cid}", file=sys.stderr)
            except Exception as exc:
                # Report failure and bubble up
                self._callback("failed", message=f"upload_failed: {exc}")
                raise

        self._callback("completed", progress=100, trained_model_cid=tm_cid)

    # --------------------------- Upload helpers ----------------------------
    def _upload_trained_model(self, file_path: Path) -> str:
        """Upload the trained model zip to backend and return trained_model_cid.

        Uses multipart/form-data over HTTP(S) with streaming to avoid loading
        the entire file into memory.
        """
        url = _ensure_https(self.ctx.upload_url) or self.ctx.upload_url
        if not url:
            raise RuntimeError("upload_url_missing")
        if not file_path.exists() or file_path.stat().st_size <= 0:
            raise RuntimeError("trained_model_zip_missing")

        # Build multipart boundaries and compute content-length
        boundary = f"---------------------------darvinrunner{int(time.time())}"
        crlf = "\r\n"

        # Prepare text fields
        fields: Tuple[Tuple[str, str], ...] = (
            ("job_id", self.ctx.job_id),
            ("job_token", self.ctx.job_token),
        )

        def _part_header(name: str, filename: Optional[str] = None, content_type: Optional[str] = None) -> bytes:
            lines = [f"--{boundary}"]
            disp = f'Content-Disposition: form-data; name="{name}"'
            if filename is not None:
                disp += f'; filename="{filename}"'
            lines.append(disp)
            if content_type:
                lines.append(f"Content-Type: {content_type}")
            lines.append("")
            return (crlf.join(lines) + crlf).encode("utf-8")

        def _part_footer() -> bytes:
            return crlf.encode("utf-8")

        # Calculate content-length
        file_size = file_path.stat().st_size
        preamble_len = 0
        for k, v in fields:
            preamble_len += len(_part_header(k)) + len(v.encode("utf-8")) + len(_part_footer())
        file_header = _part_header("trained_model", file_path.name, "application/zip")
        ending = (f"--{boundary}--" + crlf).encode("utf-8")
        content_length = preamble_len + len(file_header) + file_size + len(_part_footer()) + len(ending)

        # Open connection with retries
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise RuntimeError("invalid_upload_url")
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "darvin-runner/1.0",
            "Content-Length": str(content_length),
        }
        # retry loop
        try:
            max_retries = int(os.getenv("DARVIN_RETRY_MAX", "5"))
        except Exception:
            max_retries = 5
        try:
            base_delay = float(os.getenv("DARVIN_RETRY_BASE_DELAY", "0.5"))
        except Exception:
            base_delay = 0.5

        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= max_retries:
            if attempt > 0:
                print(f"[darvin-runner] upload retry attempt={attempt} to={parsed.scheme}://{parsed.hostname}{path}", file=sys.stderr)
            conn = conn_cls(parsed.hostname, port, timeout=300)
            try:
                conn.putrequest("POST", path)
                for hk, hv in headers.items():
                    conn.putheader(hk, hv)
                conn.endheaders()

                # Write field parts
                for k, v in fields:
                    conn.send(_part_header(k))
                    conn.send(v.encode("utf-8"))
                    conn.send(_part_footer())

                # Write file part header + content streamed
                conn.send(file_header)
                sent = 0
                with file_path.open("rb") as f:
                    while True:
                        chunk = f.read(1024 * 512)
                        if not chunk:
                            break
                        conn.send(chunk)
                        sent += len(chunk)
                conn.send(_part_footer())

                # Closing boundary
                conn.send(ending)

                resp = conn.getresponse()
                raw = resp.read()
                print(f"[darvin-runner] upload http_status={resp.status} sent_bytes={sent} content_length={content_length}", file=sys.stderr)
                # Handle redirect (rare for POST; some proxies may 307/308)
                if resp.status in (301, 302, 303, 307, 308):
                    loc = resp.getheader("Location")
                    if not loc:
                        raise RuntimeError(f"upload_redirect_no_location:{resp.status}")
                    return self._upload_trained_model_to(loc, file_path)
                if 200 <= resp.status < 300:
                    try:
                        obj = json.loads(raw.decode("utf-8", errors="ignore")) if raw else {}
                    except Exception:
                        obj = {}
                    tm_cid = ((obj.get("data") or {}).get("trained_model_cid")) if isinstance(obj, dict) else None
                    if not tm_cid:
                        raise RuntimeError("upload_no_cid_returned")
                    return str(tm_cid)
                # retry on transient
                if resp.status not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"upload_http_{resp.status}")
                last_err = RuntimeError(f"upload_http_{resp.status}")
            except Exception as e:
                last_err = e
                print(f"[darvin-runner] upload error: {e}", file=sys.stderr)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            attempt += 1
            if attempt > max_retries:
                break
            sleep_s = min(base_delay * (2 ** (attempt - 1)), 8.0) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

        if last_err:
            raise last_err
        raise RuntimeError("upload_failed")

    def _upload_trained_model_to(self, url: str, file_path: Path) -> str:
        # Helper used when following a redirect to a new absolute URL.
        self.ctx = Ctx(
            job_id=self.ctx.job_id,
            job_token=self.ctx.job_token,
            callback_url=self.ctx.callback_url,
            bundle_url=self.ctx.bundle_url,
            upload_url=url,
            output_dir=self.ctx.output_dir,
            dataset_cids=self.ctx.dataset_cids,
            hyperparams=self.ctx.hyperparams,
            subnet_address=self.ctx.subnet_address,
            work_dir=self.ctx.work_dir,
        )
        return self._upload_trained_model(file_path)

    def _safe_json(self, s: Optional[str]) -> Any:
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s

    # ------------------------------- Runner --------------------------------
    def run(self) -> int:
        def _on_term(signum, frame):  # noqa: ARG001
            try:
                self._hb_stop()
                self._callback("failed", message=f"terminated: sig={signum}")
            finally:
                os._exit(143)  # 128+15 SIGTERM

        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        try:
            bundle = self.download_bundle()
            self.run_training(bundle)
            self.produce_artifacts()
            return 0
        except Exception as exc:
            # Error already reported via callbacks where applicable
            print(f"[darvin-runner] error: {exc}", file=sys.stderr)
            return 1


def main() -> int:
    # Required envs injected by backend
    job_id = _env_required("DARVIN_JOB_ID")
    job_token = _env_required("DARVIN_JOB_TOKEN")
    callback_url = _env_required("DARVIN_CALLBACK_URL")
    # Bundle URL may be absent for local debug; in production it should exist
    bundle_url = _ensure_https(os.getenv("DARVIN_TRAIN_BUNDLE_URL"))
    upload_url = _ensure_https(os.getenv("DARVIN_UPLOAD_URL"))
    output_dir = Path(os.getenv("DARVIN_OUTPUT_DIR", "/output")).resolve()
    dataset_cids = os.getenv("DARVIN_DATASET_CIDS")
    hyperparams = os.getenv("DARVIN_HYPERPARAMS")
    subnet_address = os.getenv("DARVIN_SUBNET_ADDRESS")
    work_dir = Path(os.getenv("DARVIN_WORK_DIR", "/tmp/workspace/train/.darvin_tmp")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Ensure callback also uses https where possible to avoid proxy 301 on POST
    callback_url = _ensure_https(callback_url) or callback_url

    ctx = Ctx(
        job_id=job_id,
        job_token=job_token,
        callback_url=callback_url,
        bundle_url=bundle_url,
        upload_url=upload_url,
        output_dir=output_dir,
        dataset_cids=dataset_cids,
        hyperparams=hyperparams,
        subnet_address=subnet_address,
        work_dir=work_dir,
    )
    return Runner(ctx).run()


if __name__ == "__main__":
    sys.exit(main())
