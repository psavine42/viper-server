import zmq
import random
import sys
import time



port = "5556"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:{}".format(port))
cnt = 0

while True:
    # socket.send_string("Server message to client3")
    msg = socket.recv()

    print(msg)

    if cnt > 5:
        socket.send_string('END')
        cnt = 0
    else:
        socket.send_string("Server message to client3")
        time.sleep(1)
        cnt += 1

# print('done')

