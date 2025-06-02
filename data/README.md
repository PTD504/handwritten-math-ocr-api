**Description of the directories and files:**

* **`test_formulas/`**: This directory contains the image files (`.png` format) for the test set formulas.
* **`train_formulas/`**: This directory contains the image files (`.png` format) for the training set formulas.
* **`validate_formulas/`**: This directory contains the image files (`.png` format) for the validation set formulas.
* **`test_labels.csv`**: This CSV file provides the labels for the test set images. Each row contains the filename of an image and its corresponding LaTeX representation of the formula. The columns are: `image_filename`, `latex_label`.
* **`train_labels.csv`**: This CSV file provides the labels for the training set images. Each row contains the filename of an image and its corresponding LaTeX representation of the formula. The columns are: `image_filename`, `latex_label`.
* **`validate_labels.csv`**: This CSV file provides the labels for the validation set images. Each row contains the filename of an image and its corresponding LaTeX representation of the formula. The columns are: `image_filename`, `latex_label`.

**Note:** The image filenames in the `.csv` label files correspond to the filenames of the image files within their respective `_formulas/` directories. For example, an entry in `train_labels.csv` with `image_filename: formula_123.png` would correspond to the image file `data/train_formulas/formula_123.png`.