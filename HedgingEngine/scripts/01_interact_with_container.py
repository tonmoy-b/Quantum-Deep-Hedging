import os
import socket
import json
import time

# Configuration
DEST_IP = '127.0.0.1'
DEST_PORT = 8080

def send_test_order():
    # create udp socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # dummy input request for the hedging engine
    order_data = {
        'symbol': 'BTC-USD',
        'side': 'buy',
        'qty': 011.5,
        'price': 52000.0,
        'timestamp': time.time()
    }
    
    message = json.dumps(order_data).encode('utf-8')
    
    try:
        print(f'Sending to {DEST_IP}:{DEST_PORT}...')
        sock.sendto(message, (DEST_IP, DEST_PORT))
        print('message sent successfully, confirm in docker logs.')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        sock.close()

if __name__ == '__main__':
    send_test_order()