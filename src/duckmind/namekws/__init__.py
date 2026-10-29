"""Name keyword spotting: the duck recognizes only its own name, on-device.

Nothing here records or uploads audio. Training runs off-board (see
``notebooks/name_kws_colab.ipynb``); the robot only runs :mod:`duckmind.namekws.detector`
on 16 kHz microphone frames and reports "my name was heard" to Duck Core.
Requires numpy (and onnxruntime for detection); the rest of Duck Mind does not.
"""
