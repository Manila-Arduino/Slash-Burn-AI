# Option 1: Default route IP (often your Wi-Fi). Stdlib only.
import socket


def get_ip_default_route() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets actually sent
        return s.getsockname()[0]
    finally:
        s.close()


print(get_ip_default_route())
