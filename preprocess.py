import re
from datetime import datetime

SYSLOG_PATTERN = re.compile(
    r"^(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<service>[\w.\-/]+)\[(?P<pid>\d+)\]:\s+"
    r"(?P<message>.*)$"
)

IPV4_PATTERN = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

EVENT_TYPE_RULES = [
    ("failed password", "auth_failure"),
    ("invalid user", "invalid_user"),
    ("accepted password", "auth_success"),
    ("accepted publickey", "auth_success"),
    ("connection closed", "connection_closed"),
    ("received disconnect", "disconnected"),
    ("disconnected from", "disconnected"),
    ("reverse mapping", "suspicious"),
    ("possible break-in", "suspicious"),
]

LOG_LEVEL_BY_EVENT_TYPE = {
    "auth_failure": "warning",
    "invalid_user": "warning",
    "suspicious": "warning",
    "auth_success": "info",
    "connection_closed": "info",
    "disconnected": "info",
    "other": "info",
}


def _classify_event_type(message: str) -> str:
    lowered = message.lower()
    for keyword, event_type in EVENT_TYPE_RULES:
        if keyword in lowered:
            return event_type
    return "other"


def _extract_source_ip(message: str) -> str:
    match = IPV4_PATTERN.search(message)
    return match.group(0) if match else ""


def _normalize_timestamp(month: str, day: str, time_str: str) -> str:
    month_num = datetime.strptime(month, "%b").month
    return f"{month_num:02d}-{int(day):02d} {time_str}"


def parse_line(lineno: int, raw: str) -> dict:
    stripped = raw.strip()
    match = SYSLOG_PATTERN.match(stripped)

    if match:
        timestamp = _normalize_timestamp(match.group("month"), match.group("day"), match.group("time"))
        service = match.group("service")
        pid = match.group("pid")
        message = match.group("message")
    else:
        timestamp = ""
        service = ""
        pid = ""
        message = stripped

    event_type = _classify_event_type(message)
    log_level = LOG_LEVEL_BY_EVENT_TYPE[event_type]
    source_ip = _extract_source_ip(message) if match else ""

    return {
        "lineno": lineno,
        "raw": stripped,
        "timestamp": timestamp,
        "service": service,
        "pid": pid,
        "message": message,
        "event_type": event_type,
        "source_ip": source_ip,
        "log_level": log_level,
    }


if __name__ == "__main__":
    test_lines = [
        "Jun 14 06:32:48 server sshd[1234]: Failed password for root from 185.220.101.47 port 54321 ssh2",
        "Jun 14 06:33:01 server sshd[1234]: Accepted password for admin from 10.0.0.5 port 22 ssh2",
        "Jun 14 06:33:10 server sshd[1234]: Invalid user oracle from 192.168.1.100",
        "This line does not match syslog format at all",
    ]

    for i, line in enumerate(test_lines, start=1):
        result = parse_line(i, line)
        print(f"--- Test case {i} ---")
        for key, value in result.items():
            print(f"  {key}: {value!r}")
        print()
