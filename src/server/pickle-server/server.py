import env
import socket, pickle, time
import numpy as np

print("Server is listening on port %s..." % env.PORT)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((env.HOST, env.PORT))

while True:
    s.listen(1)
    conn, addr = s.accept()
    print("Connected to %s" % str(addr))

    # Simulate receiving buffer from bot
    data = []
    while True:
        packet = conn.recv(1024)
        if not packet: break
        data.append(packet)
    data = pickle.loads(b"".join(data))
    #print(data)
    print('Received data from client')
    print('Closing connection...\n')
    conn.close()

    conn, addr = s.accept()

    # Simulate sending weights to bot
    dummy_weights = np.random.randint(255, size=(256, 256, 3))
    dummy_weights = pickle.dumps(dummy_weights)
    print('generated dummy weights')

    start = time.time()
    conn.sendall(dummy_weights)
    end = time.time()

    print('Data sent to client in %fs' % (end - start))
    conn.close()
    print('Closing connection...\n')
