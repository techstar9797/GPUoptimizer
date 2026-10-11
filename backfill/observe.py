"""Agent observability with Raindrop.

Every generation is one Raindrop interaction. Tool calls (simulator runs, Atlas
vector searches, harness edits) are tool spans on it, and outcomes measured by
hard metrics are signals: beat_parent, regressed, beat_fgd, sandbox_violation,
duplicate_policy, guardrail_tripped. Every event carries harness_version, so
Raindrop can show how the agent's behavior changes as the harness rewrites itself.

Without RAINDROP_WRITE_KEY (or if the SDK is not installed) this is a no-op.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from .config import SETTINGS

_rd = None
if SETTINGS.raindrop_write_key:
    try:
        import raindrop.analytics as _rd  # type: ignore
        _rd.init(SETTINGS.raindrop_write_key, tracing_enabled=False)
    except Exception as e:  # missing SDK or init failure: keep running without telemetry
        print(f"Raindrop disabled: {type(e).__name__}: {e}")
        _rd = None

ENABLED = _rd is not None


class Generation:
    """One generation of the agent loop, reported to Raindrop as an interaction."""

    def __init__(self, *, event: str, harness_version: int, harness_id: str, generation: int,
                 workloads: list[str], model: str, prompt: str = ""):
        self.props = {"harness_version": harness_version, "harness_id": harness_id,
                      "generation": generation, "workloads": ",".join(workloads)}
        self.event_id = None
        self.tools: list[dict] = []
        self.signals: list[str] = []
        self._i = None
        if _rd:
            self._i = _rd.begin(user_id=SETTINGS.run_user, event=event, input=prompt[:20000],
                                properties=self.props, model=model, convo_id=f"harness-{harness_id}")
            self.event_id = getattr(self._i, "id", None)

    @contextmanager
    def tool(self, name: str, input=None, properties: dict | None = None):
        """Time a tool call and record it. Yields a dict you can put 'output' into."""
        rec = {"name": name, "output": None}
        t0 = time.time()
        err = None
        try:
            yield rec
        except Exception as e:
            err = e
            raise
        finally:
            ms = (time.time() - t0) * 1000
            self.tools.append({"name": name, "ms": round(ms), "error": str(err) if err else None})
            if self._i:
                self._i.track_tool(name=name, input=input, output=rec.get("output"),
                                   duration_ms=ms, error=err, properties=properties)

    def attach_code(self, name: str, code: str, role: str = "output") -> None:
        if self._i:
            self._i.add_attachments([{"type": "code", "name": name, "value": code[:200000],
                                      "role": role, "language": "python"}])

    def set(self, **props) -> None:
        self.props.update(props)
        if self._i:
            self._i.set_properties(props)

    def signal(self, name: str, positive: bool, **props) -> None:
        self.signals.append(name)
        if _rd and self.event_id:
            _rd.track_signal(event_id=self.event_id, name=name,
                             sentiment="POSITIVE" if positive else "NEGATIVE",
                             properties={**self.props, **props})

    def edit(self, name: str, before: str, after: str, comment: str) -> None:
        """A harness rewrite, recorded as an edit signal with the new version."""
        self.signals.append(name)
        if _rd and self.event_id:
            _rd.track_signal(event_id=self.event_id, name=name, signal_type="edit",
                             after=after[:50000], comment=comment[:2000],
                             properties={**self.props, "before": before[:20000]})

    def finish(self, output: str) -> None:
        if self._i:
            self._i.finish(output=output[:20000])


def flush() -> None:
    if _rd:
        _rd.flush()


def shutdown() -> None:
    if _rd:
        _rd.shutdown()
