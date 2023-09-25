import websocket
import cv2
import numpy as np

def on_message(ws, message):
    # Convert the received message to a numpy array
    frame = np.frombuffer(message, dtype=np.uint8)

    # Decode the frame using OpenCV
    decoded_frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Display the frame
    cv2.imshow('Received Frame', decoded_frame)
    cv2.waitKey(1)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("WebSocket closed")

def on_open(ws):
    # Read the audio file as bytes
    with open('main0.mp3', 'rb') as f:
        audio_bytes = f.read()

    # Send the audio bytes to the server
    ws.send(audio_bytes, opcode=websocket.ABNF.OPCODE_BINARY)

if __name__ == "__main__":
    # Connect to the WebSocket server
    ws = websocket.create_connection("ws://100.34.50.119:40606/ws")

    # Set up the WebSocket callbacks
    ws.on_message = on_message
    ws.on_error = on_error
    ws.on_close = on_close

    # Send the audio bytes to the server
    on_open(ws)

    # Wait for incoming messages
    while True:
        message = ws.recv()
        if not message:
            break
        on_message(ws, message)

    # Close the WebSocket connection
    ws.close()