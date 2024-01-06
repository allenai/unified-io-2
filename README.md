# Unified-IO 2
This repo contains code for [Unified-IO 2](https://unified-io-2.allenai.org/), including code to run a demo, do training,
and do inference. This codebase is modified from [T5X](https://github.com/google-research/t5x).

## Install
Install the dependencies with pip

For a TPU:
```
python3 -m pip install -e '.[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

For a GPU/CPU (note we have been using TPUs so GPU setups are not well tested):
```
python3 -m pip install -e '.' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Running the demo requires additional dependencies, install them with:
```
python3 -m pip install -e '.[demo]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

The LLaMa tokenizer also needs to be installed, download the `.model` file from https://github.com/facebookresearch/llama/tree/main?tab=readme-ov-file
and then update `t5x/examples/unified_io/config.py` so `LLAMA_TOKENIZER_PATH` points to the download location.

## Checkpoints
We make checkpoints in the T5X format available on S3:

- XXL: s3://ai2-prior-uio/public/uio2-checkpoints/xxl-3m
- XL: s3://ai2-prior-uio/public/uio2-checkpoints/xl-3m
- Large: s3://ai2-prior-uio/public/uio2-checkpoints/large-3m

To download, copy the directory recursively. For example:

```
aws s3 --no-sign-request cp --recursive s3://ai2-prior-uio/public/uio2-checkpoints/large-3m large-3m --exclude "state*"  
```

They should be copied to a local disk or to google file storage. Here, the `--exclude "state*"`
flag excludes the optimizer state from the download, it can be removed if you want
to continue training the checkpoint from the current optimizer state.

## Demo
To run the model interactively the demo notebook can be run.
Make sure the demo dependencies have been [installed](#Setup).

Then run the demo notebook:
```
jupyter notebook demo.ipynb
```


Set `FULL_CKPT_PATH` and `MODEL_TYPE` in the second cell to your checkpoint and
the correct model size. Then the notebook can be used to start the demo.

The demo shows how to load the model, parameters, and do inference.

The demo will be slow the first time it is used because the inference function
needs to be compiled, subsequent calls with similar inputs/outputs will be 
much faster.

## Data
To train and eval on entire datasets the datasets need to be registered with `seqio` in  `seqio.TaskRegistry`. See
`t5x/examples/unifiedio/data/tasks.py` for examples. See [seqio](https://github.com/google/seqio)
for more details on how datasets are managed by seqio. 
Some datasets require running a pre-processing script before they can be used.

Make sure `config.MULTITASK_TFDS_DATA_DIR` is updated to
point to the location to store the datasets.

### Datasets
We provided some initial datasets in `t5x/examples/unifiedio/data/tasks.py`. 
Our datasets are generally built one of three ways:

1. Constructed as a `tensorflow_dataset` and then uploaded to the location specified in `config.MULTITASK_TFDS_DATA_DIR`
2. Constructed as a set of tfrecords and uploaded to the same location
3. Directly using a dataset from https://www.tensorflow.org/datasets/catalog/overview

Datasets built in the first or second way require running a build script before they can be 
used. `create_data` contains the needed build scripts. For example running:

```
python3 create_data/tfdatasets/coco_all/build.py ~/data/tfds ~/data/vqa ~/data/coco_annotations
```

Will upload a tfdataset of COCO data, which allows tasks such as `image_generation_coco_2017` 
and `image_caption_coco_2017` to be used. Some datasets, such as the refexp datasets, that use 
the public tensoflow catalog might have their own manual pre-processing steps as well
which will be specified on their webpage.

UnifiedIO 2 contains a large number of tasks, for this initial release we only include
a subset but will add more as we test and verify additional tasks.

### Preprocessing
Pre-processing in UIO2 happens in three stages:

1. Task-specific pre-processing constructs a prompt and builds input and outputs in the supported modalities. This stage needs to resize and pad images into the
correct sizes, and provide masks to show which parts of the image are padding (typically with `unified_io.data.data_utils.resize_and_pad`).
Audio segments need to be converted to mel-spectrograms, which can also be masked if working with
noised data. This stage is implemented by various preprocessing functions in `unified_io.data.preprocessing`.
The demo shows how to do this for raw inputs.
To allow this stage to do different pre-processing during training and testing,
we pass a `is_training` field in sequence_length dictionary to indicate
whether the dataset is being used for training or testing. 
2. Next `modality_processing.unified_io_preprocessor` is run. This function does various task-general pre-preprocessing steps, 
such as tokenizing the text, and adds empty values for missing modalities so the output dataset has a consistent set of fields.
3. Finally `UnifiedIOFeatureConverter` is applied, this can happen
after multiple datasets have been combined into a `seqio.Mixture`. 
This function will make sure the output dataset has a consistent structure and is padded to have
fixed-size tensors, as is needed for jax. This dataset can now be batched and passed directly
into the loss or prediction functions of a UnifiedIO 2 model.
The padding is determined by the sequence_len dictionary.
   
To add a dataset, register it with seqio and ensure the last pre-processor
is `modality_processing.unified_io_preprocessor`. The preceding
functions should make sure the dataset has the appropriate fields for that function.

### Prompts
Our entire set of prompts in `t5x/examples/unified_io/data/prompt_dict`,
we randomly select among these prompts during training.


### Visualization
We include a visualization script to show what the data looks like after post-processing:

```
python3 t5x/examples/unified_io/scripts/dataset_visualize.py refcoco_unc viz --override```
```

To get a more compact view:

```
python3 t5x/examples/unified_io/scripts/dataset_visualize.py refcoco_unc viz --override --gin.get_target_modalities.target_modality=[\"text\"] --gin.get_input_modalities.input_modality=[\"text\",\"image\"] --nomasks
```

## Training
Once a checkpoint is downloaded and a dataset is ready, training can be run using train.py.
Our training strategy largely follows T5X, which is configured through [gin](https://github.com/google/gin-config).
Follow the setup from `https://github.com/google-research/t5x` to train on TPUs.

For example, to fine-tune the large model on refexp:

```
python3 t5x/train.py --gin_file=t5x/examples/unified_io/t5_1_1/large.gin --gin_file=t5x/examples/unified_io/t5_1_1/finetune/refexp.gin --gin.INITIAL_CHECKPOINT_PATH=\"/path/to/checkpoint\" --gin.MODEL_DIR=\"path/to/output_dir\" --gin.BATCH_SIZE=8
```

### Modalities
UnifiedIO 2 can be run on a subset of the supported modality, which makes training more 
efficient. This can be set through the gin-configured parameters in 
`get_input_modalities` and `get_target_modalities`. For example, refexp.gin 
only turns on the image/text inputs and text outputs.

### Sequence Lengths
Due to jax's fixed size tensor constraint, we by default pad all inputs and targets to the 
model to the maximum length supported. When training on mixtures where this is excessive,
this can be tweaked by changing the sequence_lengths used by `seqio` 
For example, refexp,gin reduce the input and output sequence length since 
refexp has little text.

### Wandb
We have modified train.py to use wandb, just make sure a `WANDB_API_KEY` environment variable is set. 
The gin configurable function `utils.init_wandb` should be modified or configured
through gin to select the correct name/group/project/entity.

### Packing
If the training mixture contains a mix of long and short examples, packing
can make things more efficient. Packing will pack up to two examples together
into a single input sequence, it can be turned on with this flag:

```
--gin.PackingStrategy.pack_max_len=(864, 1280)
```

During training, two examples will be attempted to be packed in a sequence with total
input length of 864 input length and target length or 1280. A heuristic algorithm
will try to find pairs of examples that fit this criterion as data is streamed to 
the training server, if none are found only one example will be used. 
If this happens too frequently it is a good idea to increase the max length.
Statistics will be logged to wandb to track the packing efficiency.

## Evaluation
Evaluation script are run using eval.py, for example:

```
python3 t5x/eval.py --gin_file=t5x/examples/unified_io/t5_1_1/large.gin --gin_file=t5x/examples/unified_io/t5_1_1/eval/vision_language.gin --gin.CHECKPOINT_PATH=\"large-3m\" --gin.MIXTURE_OR_TASK_NAME=\"refcoco_unc\" --gin.EVAL_OUTPUT_DIR=\"output\"
```

The target dataset must have metrics registered with seqio. Evaluations script
can be similarly made more efficient by only using the needed modalities and
choosing the sequence lengths appropriately. Note most of our official results
come from collecting outputs and then running offline evaluations, the metrics
here are used mostly for validation scores.

## Citation