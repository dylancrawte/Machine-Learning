# Welcome to my machine learning notebooks

## Please note

These notebooks train image models with PyTorch. That work is much faster on a **GPU** than on a CPU alone.

You *can* run them locally if you have a machine with a CUDA-capable NVIDIA GPU and the right drivers installed. On a typical laptop without a GPU, training can take a very long time or feel stuck.

**Recommended: run them on [Kaggle Notebooks](https://www.kaggle.com/code).** Kaggle gives you free access to GPU time in the browser, and the notebooks in this repo are written to use Kaggle paths and datasets out of the box.

### Why use Kaggle instead of running locally?

| | Local (no GPU) | Local (with GPU) | Kaggle Notebook |
|---|---|---|---|
| Training speed | Very slow | Fast | Fast (free GPU) |
| Setup | Install PyTorch, CUDA, datasets yourself | Same, plus GPU drivers | Mostly click-through |
| Dataset paths | You must download and fix paths | Same | Add dataset → paths work |

The card classifier notebook, for example, loads images from `/kaggle/input/...` and uses a pretrained model (`EfficientNet-B0` via `timm`). That combination is practical on Kaggle’s GPU; on CPU-only hardware it is still possible but not a good first experience.

### How to set up a Kaggle notebook

1. **Create a Kaggle account**  
   Sign up at [kaggle.com](https://www.kaggle.com) if you do not have one.

2. **Open (or create) a notebook**  
   - To start from this repo: copy the notebook into Kaggle, or upload `card-classifier/model-card-classifier.ipynb` via **Code → New Notebook → File → Upload notebook**.  
   - Or fork an existing public notebook that uses the same dataset.

3. **Turn on the GPU**  
   - In the notebook editor: **Settings** (gear icon on the right).  
   - Under **Accelerator**, choose **GPU** (not “None”).  
   - Save. Kaggle will restart the session with GPU enabled when needed.

4. **Add the playing cards dataset**  
   The card classifier expects the [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification) dataset.  
   - In the notebook sidebar, click **+ Add data** (or **Add Input**).  
   - Search for `cards-image-datasetclassification` or open the dataset page and click **New Notebook** / **Add to notebook**.  
   - After it is attached, files appear under `/kaggle/input/cards-image-datasetclassification/` with `train/`, `valid/`, and `test/` folders.

5. **Run the notebook**  
   - Use **Run All** or run cells top to bottom.  
   - The first run may download pretrained weights (`timm` / EfficientNet); that is normal.  
   - If a cell references `/kaggle/input/...`, do not change those paths unless you are using a different dataset layout.

6. **Save your work**  
   - Kaggle autosaves the notebook to your account.  
   - To keep outputs or a trained model, download artifacts from the notebook or add cells to save files under `/kaggle/working/` (Kaggle’s writable folder).

### Quick checklist before you train

- [ ] Accelerator set to **GPU** in notebook settings  
- [ ] **Cards Image Dataset-Classification** added as an input  
- [ ] Notebook paths point at `/kaggle/input/cards-image-datasetclassification/...`  
- [ ] Internet enabled if the notebook must download pretrained weights (check **Settings → Internet** if downloads fail)

For a step-by-step explanation of what the card classifier notebook does (dataset, model, training loop, predictions), see [card-classifier/README.md](card-classifier/README.md).