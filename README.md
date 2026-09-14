This is a Backpropogated Neural Network trained on `sklearn.datasets.load_digits` which are 8x8 greyscale images of digits

This project will:
- Load and normalize the dataset
- Train an NN (64 > 32 > 10) using manually implemented backpropogation
- Print train/test loss and accuracy
- Save the loss curve to `outputs/loss_curve.png` and the summary to `outputs/results.txt`

Steps to Run:

- Run: python3 -m venv .venv

- Run: source .venv/bin/activate

- Run: pip install -r requirements.txt

- Run: python -m pytest tests/test_gradients.py -v -s

- Run: python train.py

The project is mostly coded by me, but the frameworks and initial setup I had to do with AI since I was having difficulty understanding how to arrange the files and set everything up pretty much. Most of the math part is from my own understanding (which I did have issues with so I had Claude fix it). Also had it do a bit of bugfixing when I got stuck. I made the project on Google Colab and shifted everything to a repo so I tried to make commits in the order I made them on the Colab file.