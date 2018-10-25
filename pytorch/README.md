The following steps guide the user to run the pytorch
implementation of CNN based spoofing detector for the
ASVSpoof2017 dataset.

# Software requirements

The recipe has been tested to run on Debian 8.9.
It requires:

1. Matlab, incl. Signal Processing Toolkit
2. miniconda environment: see ``INSTALL`` to setup pytorch environment

It does not require a GPU. It can be run on a CPU.

# Step 0 - Data preparation

Three file lists need to be prepared for the rest of the script
to run without errors

1. ``filelist_train``: contains the list of all train files
2. ``filelist_dev``: contains the list of all development files
3. ``filelist_eval``: contains the list of all evaluation files

Each entry in the list is the path  to the audio (wav) file.
The audio files are those distributed with the dataset.

# Step 1 - Extract features in Matlab

Run ``run_normallogpow_V2.m``. The files are saved in the current
directory. To change the destination directory change the basedir
variable in the code.

One way to run the Matlab script from commandline is 

```bash
matlab -nodisplay -nodesktop -r "run_normallogpow_V2.m"
```


# Step 2 - Split features into separate files

Run ``split-data-768.py`` from the current directory.
The split files will be saved in ``data/768/``.

If the destination directory in Step 1 was changed, pass
this value to ``python split-data-768.py .``

```bash
python split-data-768.py destdir
```

# Step 3 - Train and Evaluate

The pytorch code assumes that pytorch is installed in the environment.
If it was installed with conda or pip, load the environment as appropriate.

It also assumes that the data is present in ``data/768`` and the code is run from
the current directory.

To train and test, run

```bash
source path-to-miniconda/bin/activate pytorch
bash run.sh
```

The system outputs for the dev set will be available in a file called 
``dev-eer`` for the development set and eer-file for the eval set.
Use Kaldi's compute-eer to get the Equal Error Rate. For example:

```bash
compute-eer eer-file
```
