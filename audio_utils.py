import os, time
from io import BytesIO

import numpy as np
import pandas as pd
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pygame


# method to measure time 
def timeit(method: object) -> object:
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print (f"{method.__name__} {(te - ts):2.2f} s")
        return result
    return timed


class AudioRecorder():
    def __init__(self) -> None:
        self.mic = sr.Microphone()
        self.recognizer = sr.Recognizer()
    
    @timeit
    def record_audio(self, adjust_for_ambient_noise: bool=False) -> sr.AudioData:
        with self.mic as source:
            if adjust_for_ambient_noise:
                audio = self.recognizer.adjust_for_ambient_noise(source)
            else:
                audio = self.recognizer.listen(source)
        return audio
        
        
class SpeechRecognizer():
    def __init__(self, language: str="en_US") -> None:
        self.language = language
        self.recognizer = sr.Recognizer()
    
    @timeit
    def recognize(self, audio_input: sr.AudioData) -> sr.AudioData:
        text_output = self.recognizer.recognize_google(audio_input,language=self.language)
        return text_output

    
class TextToAudioConverter():
    def __init__(self, language: str="en") -> None:
        self.language = language
    
    @timeit
    def output_audio(self, input_text: str) -> None:
        tts = gTTS(text=input_text,lang=self.language)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)