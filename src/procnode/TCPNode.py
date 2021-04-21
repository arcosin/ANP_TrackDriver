from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
from collections import defaultdict
import socket

class TCPNode():
    def __init__(self, object, host, port):
        self.portno = port
        self.host = host
        self.object = object

    def getData(self):
        return self.object

    def setupServer(self):
        self.server = SimpleXMLRPCServer((self.host, self.portno))
        self.server.register_function(self.getData, 'getData')

    def send(self, forever=False):    
        if forever:
            self.server.serve_forever()
        else:
            self.server.handle_request()
        
    def setupClient(self):
        self.proxy = xmlrpc.client.ServerProxy("http://" + self.host + ":" + str(self.portno) + "/")

    def recv(self, block = True):
        return self.proxy.getData()
