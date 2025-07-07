import time

def timer(countdown: int = 10) -> None:
    """Countdown timer."""
    print("Starting in...")
    for t in range(countdown, 0, -1):
        print(t, end=' ')
        time.sleep(1)
    print()