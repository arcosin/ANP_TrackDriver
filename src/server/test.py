import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '..'))
from procnode import *

if __name__ == "__main__":
    print("Start test")
    toSend = dict({"I get": "Nothing", "You get": "This stupid meme"})
    typeOf = "TCP"
    id_s = "A"

    s = TCPNode(id_s, toSend, typeOf, 25565)
    s.serverStart()