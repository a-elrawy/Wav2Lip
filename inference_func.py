import os
import numpy as np
import cv2
import argparse
import audio
import json
import subprocess
import time
from tqdm import tqdm
import torch
import face_detection
from models import Wav2Lip
import platform
import pickle

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

# Load the model in memory

model_path = 'wav2lip_gan.pth'
model = load_model(model_path)
face_det_filename = "face_det_results.pkl"
# Load preprocessed face detection results from a .npy file
with open(face_det_filename, "rb") as f:
    face_det_results = pickle.load(f)

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

def face_detect_from_pkl(pkl_file_path="face_det_results.pkl"):
    with open(pkl_file_path, "rb") as f:
        face_detections = pickle.load(f)
    return face_detections

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images, face_det_batch_size, pads, nosmooth, wav2lip_batch_size, precomputed_detections=None):
    if precomputed_detections:
        return face_detect_from_pkl(precomputed_detections)

    batch_size = face_det_batch_size

    batch_size = face_det_batch_size
    
    while 1:
        predictions = []
        try:
            start = time.time()
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
                print(f"Time taken for face detection: {time.time() - start}")
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results

def datagen(frames, mels, img_size,static,wav2lip_batch_size, face_det_filename="face_det_results.pkl"):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    
    start = time.time()
    for i, m in enumerate(mels):
        idx = 0 if static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            print(f"Time taken for datagen: {time.time() - start}")

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
            
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        print(f"Time taken for datagen: {time.time() - start}")
        yield img_batch, mel_batch, frame_batch, coords_batch

    print(f"Time taken for datagen: {time.time() - start}")
    


def lip_sync(face, audio_path, outfile, fps=30, resize_factor=1, rotate=False, crop=[0, -1, 0, -1],
             img_size=96, wav2lip_batch_size=16, face_det_batch_size=16, pads=[0, 10, 0, 0],
             static=False, nosmooth=False):
    if not os.path.isfile(face):
            raise ValueError('--face argument must be a valid path to video/image file')
    
    elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face)]
        fps = fps

    else:
        start = time.time()
        video_stream = cv2.VideoCapture(face)
        print(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        print("Video FPS: {}".format(fps))
        
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print(f"Time taken for reading video: {time.time() - start}")
    print ("Number of frames available for inference: "+str(len(full_frames)))

    start = time.time()
    if not audio_path.endswith('.wav'):
        print('Extracting raw audio_path...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'
    print(f"Time taken for extracting audio: {time.time() - start}")
    start = time.time()
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    print(f"Time taken for mel spectrogram: {time.time() - start}")
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks, img_size, static, batch_size, f"face_det_results.pkl")
    
    start = time.time()
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi', 
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)
    print(f"Time taken for lip sync: {time.time() - start}")
    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')



