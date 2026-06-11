from collections import Counter
from datetime import datetime

BURST_EVENT_TYPES = ("auth_failure", "invalid_user")

SEVERITY_ORDER = {"warning": 2, "info": 1}
SEVERITY_TO_LEVEL = {2: "warning", 1: "info", 0: "info"}


def _severity(level: str) -> int:
    return SEVERITY_ORDER.get(level, 0)


def _dominant_value(values: list[str], fallback: str) -> str:
    non_empty = set(v for v in values if v)
    if not non_empty:
        return fallback
    if len(non_empty) == 1:
        return next(iter(non_empty))
    return "mixed"


def _dominant_event_type(values: list[str]) -> str:
    counts = Counter(values)
    ranked = counts.most_common()
    if len(ranked) == 1:
        return ranked[0][0]
    if ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return "mixed"


def _time_delta_seconds(ts1: str, ts2: str) -> float:
    if not ts1 or not ts2:
        return 0
    t1 = datetime.strptime(ts1, "%m-%d %H:%M:%S")
    t2 = datetime.strptime(ts2, "%m-%d %H:%M:%S")
    return abs((t2 - t1).total_seconds())


def _build_chunk(lines: list[dict], source: str, chunk_type: str) -> dict:
    max_severity = max(_severity(l["log_level"]) for l in lines)

    return {
        "text": "\n".join(l["raw"] for l in lines),
        "start_line": lines[0]["lineno"],
        "end_line": lines[-1]["lineno"],
        "chunk_size": len(lines),
        "source": source,
        "log_level": SEVERITY_TO_LEVEL[max_severity],
        "service": _dominant_value([l["service"] for l in lines], ""),
        "source_ip": _dominant_value([l["source_ip"] for l in lines], ""),
        "event_type": _dominant_event_type([l["event_type"] for l in lines]),
        "chunk_type": chunk_type,
        "timestamp_start": lines[0]["timestamp"],
        "timestamp_end": lines[-1]["timestamp"],
    }


def chunk_lines(
    parsed_lines: list[dict],
    source: str,
    burst_window_seconds: int = 120,
    time_window_seconds: int = 60,
    max_chunk_size: int = 30,
) -> list[dict]:
    # Pass 1: IP burst detection
    open_bursts: dict[str, list[dict]] = {}
    burst_chunks: list[dict] = []
    claimed: set[int] = set()

    for line in parsed_lines:
        if line["event_type"] not in BURST_EVENT_TYPES or not line["source_ip"]:
            continue

        ip = line["source_ip"]

        if ip not in open_bursts or not open_bursts[ip]:
            open_bursts[ip] = [line]
            continue

        burst = open_bursts[ip]
        delta = _time_delta_seconds(burst[0]["timestamp"], line["timestamp"])

        if delta <= burst_window_seconds and len(burst) < max_chunk_size:
            burst.append(line)
            if len(burst) >= max_chunk_size:
                burst_chunks.append(_build_chunk(burst, source, "ip_burst"))
                claimed.update(l["lineno"] for l in burst)
                open_bursts[ip] = []
        else:
            if len(burst) >= 2:
                burst_chunks.append(_build_chunk(burst, source, "ip_burst"))
                claimed.update(l["lineno"] for l in burst)
            open_bursts[ip] = [line]

    for burst in open_bursts.values():
        if len(burst) >= 2:
            burst_chunks.append(_build_chunk(burst, source, "ip_burst"))
            claimed.update(l["lineno"] for l in burst)

    # Pass 2: time-window chunking for unclaimed lines
    time_chunks: list[dict] = []
    accumulator: list[dict] = []

    for line in parsed_lines:
        if line["lineno"] in claimed:
            continue

        if not accumulator:
            accumulator.append(line)
            continue

        if len(accumulator) >= max_chunk_size:
            time_chunks.append(_build_chunk(accumulator, source, "time_window"))
            accumulator = [line]
            continue

        last = accumulator[-1]
        if last["timestamp"] and line["timestamp"]:
            delta = _time_delta_seconds(last["timestamp"], line["timestamp"])
            if delta > time_window_seconds:
                time_chunks.append(_build_chunk(accumulator, source, "time_window"))
                accumulator = [line]
                continue

        accumulator.append(line)

    if accumulator:
        time_chunks.append(_build_chunk(accumulator, source, "time_window"))

    all_chunks = burst_chunks + time_chunks
    all_chunks.sort(key=lambda c: c["start_line"])
    return all_chunks


if __name__ == "__main__":
    burst_timestamps = [
        "06-14 06:32:48",
        "06-14 06:32:53",
        "06-14 06:32:58",
        "06-14 06:33:04",
        "06-14 06:33:10",
    ]
    burst_lines = []
    for i, ts in enumerate(burst_timestamps):
        lineno = 66 + i
        burst_lines.append({
            "lineno": lineno,
            "raw": f"Jun 14 {ts[3:]} server sshd[1234]: Failed password for root from 185.220.101.47 port 5432{i} ssh2",
            "timestamp": ts,
            "service": "sshd",
            "pid": "1234",
            "message": f"Failed password for root from 185.220.101.47 port 5432{i} ssh2",
            "event_type": "auth_failure",
            "source_ip": "185.220.101.47",
            "log_level": "warning",
        })

    normal_specs = [
        ("06-14 06:35:00", "auth_success"),
        ("06-14 06:35:08", "connection_closed"),
        ("06-14 06:35:15", "connection_closed"),
    ]
    normal_lines = []
    for i, (ts, event_type) in enumerate(normal_specs):
        lineno = 100 + i
        normal_lines.append({
            "lineno": lineno,
            "raw": f"Jun 14 {ts[3:]} server sshd[5678]: {event_type} for admin from 10.0.0.5",
            "timestamp": ts,
            "service": "sshd",
            "pid": "5678",
            "message": f"{event_type} for admin from 10.0.0.5",
            "event_type": event_type,
            "source_ip": "10.0.0.5",
            "log_level": "info",
        })

    parsed_lines = burst_lines + normal_lines
    chunks = chunk_lines(parsed_lines, source="test.log")

    for chunk in chunks:
        print(
            f"chunk_type={chunk['chunk_type']!r} "
            f"start_line={chunk['start_line']} "
            f"end_line={chunk['end_line']} "
            f"chunk_size={chunk['chunk_size']} "
            f"source_ip={chunk['source_ip']!r} "
            f"event_type={chunk['event_type']!r}"
        )

    assert len(chunks) == 2, f"expected 2 chunks, got {len(chunks)}"
    assert chunks[0]["chunk_type"] == "ip_burst", f"expected first chunk to be ip_burst, got {chunks[0]['chunk_type']}"
    assert chunks[0]["source_ip"] == "185.220.101.47", f"expected source_ip 185.220.101.47, got {chunks[0]['source_ip']}"
    assert chunks[1]["chunk_type"] == "time_window", f"expected second chunk to be time_window, got {chunks[1]['chunk_type']}"

    print("All assertions passed.")
