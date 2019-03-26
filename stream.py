# Message Sender
import os
from socket import *


def client(args, kwargs):
    print("args", args)
    host = "127.0.0.1"  # set to IP address of target computer
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    data = str.encode("client connected")
    while True:
        UDPSock.sendto(data, addr)
        data = str.encode(input("Enter message to send or type 'exit': "))
        if data == "exit":
            break
    UDPSock.close()
    os._exit(0)


def server():
    host = ""
    port = 13000
    buf = 1024
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    UDPSock.bind(addr)
    print("Waiting to receive messages...")

    while True:
        (data, addr) = UDPSock.recvfrom(buf)
        print("Received message: " + str(data))
        if data == "exit":
            break

    UDPSock.close()
    os._exit(0)