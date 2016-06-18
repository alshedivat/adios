# ADIOS: Architectures Deep In Output Space

ADIOS is implemented as a thin wrapper around Keras' `Graph` model (i.e., multiple-input multiple-output deep architecture) by adding the adaptive thresholding functionality as described in the paper.
`adios.utils.assemble.assemble` helper function provides and handy way to construct ADIOS and MLP models from config dictionaries.
Configs can be generated from templates using `adios.utils.jobmab.gen_configurations`.
Examples of templates are given in `configs/` folder in YAML format.
Additionally, we provide utility functions for hyperparameter or architecture search using [Jobman](http://deeplearning.net/software/jobman/about.html).

All example scripts are given in `scripts/`.

**Note:** `keras.models.Graph` is no longer supported starting from `keras-v1.0` as of April, 2016.
The current version of ADIOS uses the legacy code, `keras.legacy.models.Graph`.


### Requirements
- `NumPy`
- `pyyaml`
- `Theano`
- `keras>=1.0`
- `scikit-learn`

The requirements can be installed via `pip` as follows:

```bash
$ pip install -r requirements.txt
```

Optional (needed only for using Jobman):
- `argparse`
- [Jobman](http://deeplearning.net/software/jobman/about.html)


### Installation
To use the code, we recommend installing it as Python package in the development mode as follows:

```bash
$ python setup.py develop [--user]
```

The `--user` flag (optional) will install the package for a given user only.


### Citation policy
If you use this code (in full or in part) for academic purposes, please cite our paper:

```bibtex
@inproceedings{cisse2016adios,
  title={ADIOS: Architectures Deep In Output Space},
  author={Cisse, Moustapha and Al-Shedivat, Maruan and Bengio, Samy},
  booktitle={Proceedings of The 33rd International Conference on Machine Learning},
  pages={2770-–2779},
  year={2016}
}
```

### License

MIT (for details, please refer to [LICENSE](https://github.com/alshedivat/adios/blob/master/LICENSE))

Copyright (c) 2016 Moustapha Cisse, Maruan Al-Shedivat, Samy Bengio
