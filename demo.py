from typing import Generator

from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import logging
import shutil
import torch
import glob
import cv2
import sys
import os


def get_video_frames_generator(source_path: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    success, frame = video.read()
    while success:
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = video.read()
    video.release()


def normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def caption(SOURCE_VIDEO_PATH, SOURCE_NAME, vis_processors, device, model, model_extractor):
    logging.info('Starting %s', SOURCE_NAME)
    ### Get caption for every second (24 frames)
    captions = []
    frames = []
    indices = []
    
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

    for index, frame in enumerate(generator):
        try:
            if index % 24 == 0:
                image = vis_processors["eval"](frame).unsqueeze(0).to(device)
                caption = model.generate({"image": image})[0]

                captions.append(caption)
                frames.append(frame)
                indices.append(index)
        except:
            pass

    ### Get embeddings for every caption
    sample = {"text_input": captions}
    features_text = model_extractor.extract_features(sample, mode="text")
    features_text = features_text.text_embeds_proj[:, 0]
    
    ### Find duplicate neighboring captions using embeddings
    indices_flag = []
    embeddings = []
    for x in range(len(features_text)):
        embedding_current = normalize(features_text[x].cpu().detach().numpy().flatten().reshape(1,-1))
        embeddings.append(features_text[x].cpu().detach().numpy())
        
        if x == 0: 
            indices_flag.append(True)
            continue
        if x == len(features_text)-1:
            indices_flag.append(False)
            break
        
        embedding_previous = normalize(features_text[x-1].cpu().detach().numpy().flatten().reshape(1,-1))
        similarity = np.dot(embedding_current, embedding_previous.T)
        
        if similarity >= 0.85:
            indices_flag.append(False)
        else:
            indices_flag.append(True)

    ###  Remove duplicate neighboring embeddings
    captions_array = np.asarray(captions)
    indices_flag_array = np.asarray(indices_flag)
    indices_array = np.asarray(indices)
    frames_array = np.asarray([np.asarray(i) for i in frames])
    embeddings_array = np.asarray(embeddings)
    
    captions_array = captions_array[indices_flag_array]
    frames_array = frames_array[indices_flag_array]
    indices_array = indices_array[indices_flag_array]
    embeddings_array = embeddings_array[indices_flag_array]
    
    return embeddings_array, frames_array, captions_array


def make_dir(folder=""):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        
    if not os.path.exists(folder):
        os.mkdir(folder)
        

def main():
    make_dir("output")
    SOURCE_VIDEO_PATHS = glob.glob('input/*.mp4')
    SOURCE_VIDEO_PATHS = [os.path.normpath(i) for i in SOURCE_VIDEO_PATHS]
    SOURCE_FILE_NAMES = [i.split(os.sep)[1] for i in SOURCE_VIDEO_PATHS]
    SOURCE_NAMES = [i.split('.mp4')[0] for i in SOURCE_FILE_NAMES]

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    ### This is the model for captioning
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    vis_processors.keys()

    ### This is the model for extracting embeddings
    model_extractor, vis_processors_extractor, txt_processors_extractor = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )
    
    for x, i in enumerate(SOURCE_VIDEO_PATHS):
        
        make_dir(f"output/{SOURCE_NAMES[x]}")
        
        embeddings, frames_array, captions_array = caption(i, SOURCE_NAMES[x], vis_processors, device, model, model_extractor)
        np.save(f"output/{SOURCE_NAMES[x]}/{SOURCE_NAMES[x]}_embeddings", embeddings)
        np.save(f"output/{SOURCE_NAMES[x]}/{SOURCE_NAMES[x]}_captions", captions_array)

        for y, i in enumerate(frames_array):
            im = Image.fromarray(i)
            im.save(f"output/{SOURCE_NAMES[x]}/{SOURCE_NAMES[x]}_{y}.png")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, force=True,
        format='[%(asctime)s] [%(levelname)s] {%(filename)s:%(lineno)d} - %(message)s',
        handlers=[logging.FileHandler("log.log", "w"),
                logging.StreamHandler(sys.stdout)]
    )
    
    main()
    
    logging.info('FINISHED')
