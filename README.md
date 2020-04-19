# Transformers from Scratch

Quick implementation of transformers from scratch. This will be based on the blog by Peter Bloem available [here](http://www.peterbloem.nl/blog/transformers) with a few minor changes.

# Installation

Download the project and install the requirements as follows:
```py
git clone git@github.com:goodhamgupta/transformers_from_scratch.git
cd transformers_from_scratch/
pip3 install -r requirements.txt
```

To run the classification transformer on the IMDB reviews dataset, execute the following command:

# Experiments

## Classification Transformer

To run the classification transformer, execute:
```py
python3 experiments/classify.py
```

This command will:

- Download the IMDB dataset
- Create the training and validation/test datasets depending on the config options specified
- Create a NN containing a word embedding, position embedding, bunch of transformer blocks, and a final softmax layer for the prediction.
- Train this NN on the dataset. Record the training loss and write the metrics to the `runs/` directory, which can be visualized on tensorboard using the command: `tensorboard --logdir=runs/`.

