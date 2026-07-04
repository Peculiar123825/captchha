import time

ATTEMPT_LIMIT = 5
TIME_WINDOW = 60  # seconds

attempts = {}


def check_rate_limit(ip):
    """
    Returns (allowed, current_count).
    allowed is False if this IP has exceeded ATTEMPT_LIMIT
    attempts within the last TIME_WINDOW seconds.
    """
    now = time.time()
    if ip not in attempts:
        attempts[ip] = []

    # keep only attempts within the time window
    attempts[ip] = [t for t in attempts[ip] if now - t < TIME_WINDOW]
    attempts[ip].append(now)

    if len(attempts[ip]) > ATTEMPT_LIMIT:
        return False, len(attempts[ip])
    return True, len(attempts[ip])
