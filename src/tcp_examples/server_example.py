import sys
import socket, pickle, time
import numpy as np
sys.path.append("..")
from procnode import TCPNode

server = TCPNode("server-node",True)
server.start()  
while True:

    # Simulate receiving buffer from bot
    data = server.recv(False)
    if data == None:
        break
    print(data)
    print('Received data from client')
    
    # Simulate sending weights to bot
    dummy_weights = np.random.randint(255, size=(256, 256, 3))
    print('generated dummy weights')

    start = time.time()
    server.send("client-node",dummy_weights)
    end = time.time()

    print('Data sent to client in %fs' % (end - start))
