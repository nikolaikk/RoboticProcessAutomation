# RoboticProcessAutomation
Is is a test version of Robotic Process Automation project. It has features of speech recognition and its interpretation and further mapping them onto supported activities/tasks. Mapping is done using machine learning algorithms, for now it is a dense neural net for the classification and BERT tokenizer. There are not that many tasks supported since this is supposed to be a demo, all supported tasks can be seen in `rpa_commands.py`.


Requirements:
1. Install pyaudio, the instructions can be found here for Windows [here](https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3) and [here](https://stackoverflow.com/questions/33851379/pyaudio-installation-on-mac-python-3) for MacOS.
2. Install all requirements from requirements.txt by running `pip install -r requirements.txt`.
3. In order to run main predict file in jupyter lab install ipywidgets `pip install ipywidgets` and enable them `jupyter nbextension enable --py widgetsnbextension
`.
4. Download and unzip model weights into current project, weights can be found [here](https://drive.google.com/file/d/1rkeu4Gkf7hzC2wHqqUbuKoJf8JEkeiG5/view?usp=sharing)


Instructions:
Now you are ready to run the main jupyter notebook scripts.
1. Open `predict.ipynb` in jupyter lab.
2. Run the cells consequentially.
3. As an example, you can RPA to send an email.
