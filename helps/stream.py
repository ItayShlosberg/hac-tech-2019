# Message Sender
import os
from socket import *
import socket

def client(args, kwargs):
    print("start client...")
    print("args", args)
    host = "23.96.217.227"  # set to IP address of target computer
    port = 22
    addr = (host, port)
    UDPSock = socket.socket(AF_INET, SOCK_DGRAM)
    data = str.encode("client connected")
    while True:
        UDPSock.sendto(data, addr)
        cmd = input("Enter message to send or type 'exit': ")
        data = str.encode(cmd)
        if data.decode("utf-8") == "exit":
            break
    UDPSock.close()
    os._exit(0)


def server(i,j):
    print("start server...")
    host = ""
    port = 1024
    buf = 1024
    addr = (host, port)
    UDPSock = socket.socket(AF_INET, SOCK_DGRAM)
    UDPSock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # UDPSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    UDPSock.bind(addr)
    print("Waiting to receive messages...")

    while True:
        (data, addr) = UDPSock.recvfrom(buf)
        print("Received message: " + str(data))
        if data == "exit":
            break

    UDPSock.close()
    os._exit(0)

