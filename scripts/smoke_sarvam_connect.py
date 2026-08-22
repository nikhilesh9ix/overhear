"""Validate the Sarvam key, URL and query params by opening a real session.

Sends a short burst of real (silent) PCM so the server has something to chew on,
then waits for session.begin. This proves auth + protocol without needing a
recorded WAV; interim-transcript behaviour is proven separately by
smoke_sarvam.py once a WAV exists.
"""
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.core.sarvam import SarvamStream, build_url  # noqa: E402


async def main() -> int:
    print(f"url: {build_url()}", flush=True)
    print(f"key: {'set (' + str(len(settings.sarvam_api_key)) + ' chars)' if settings.sarvam_api_key else 'MISSING'}", flush=True)

    silence = b"\x00\x00" * int(settings.sarvam_sample_rate * 0.2)  # 200ms
    try:
        async with SarvamStream() as s:
            print(f"[{s.elapsed_ms:7.1f}ms] connected", flush=True)

            async def pump():
                for _ in range(5):
                    await s.send_audio(silence)
                    await asyncio.sleep(0.2)
                await s.finish()

            task = asyncio.create_task(pump())
            got_begin = False
            try:
                async with asyncio.timeout(20):
                    async for ev in s.events():
                        shown = repr(ev.text) if ev.text else ""
                        print(f"[{ev.at_ms:7.1f}ms] {ev.kind:14s} {shown} {ev.raw}", flush=True)
                        if ev.kind == "session_begin":
                            got_begin = True
                        if ev.kind == "end":
                            break
                        if ev.kind == "error" and (ev.raw or {}).get("is_fatal"):
                            print("!! fatal error from Sarvam", flush=True)
                            return 1
            except asyncio.TimeoutError:
                print("!! timed out waiting for events", flush=True)
            task.cancel()

            if got_begin:
                print("\n===== SARVAM AUTH + PROTOCOL OK =====", flush=True)
                return 0
            print("\n!! no session.begin received", flush=True)
            return 1
    except Exception as e:
        print(f"!! {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
