from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
from collections import defaultdict
import socket

class TCPNode():
    def __init__(self, id, object, type, host, port):
        #super().__init__(id)
        self.portno = port
        self.type = type
        self.host = host
        self.object = object

    def reader(self):
        return self.object

    def createProxy(self):
        self.server.register_function(self.reader, 'reader')

    def serverStart(self):    
        self.server = SimpleXMLRPCServer((self.host, self.portno))
        self.createProxy()
        print("Listening on port " + str(self.portno) + " ...")
        self.server.handle_request()

    def startTCPClient(self):
        self.proxy = xmlrpc.client.ServerProxy(self.host + ":" + str(self.portno))

    def recv(self, block = True):                                               #TODO: Implement for super version and tcp nodes.
        print("IN NODE :" + str(self.id))
        flag = 0
        if self.type == "TCP":
            self.proxy = xmlrpc.client.ServerProxy(self.host + ":" + str(self.portno))
            print(self.proxy.reader())
        else:
            super().recv(block = block)










#===============================================================================
