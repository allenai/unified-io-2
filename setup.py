# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install T5X."""

import os
import sys
import setuptools

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 't5x')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

_jax_version = '0.2.27'
_jaxlib_version = '0.1.76'

setuptools.setup(
    name='t5x',
    version=__version__,
    description='UnifiedIO 2',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license='Apache 2.0',
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.gin'],  # not all subdirectories may have __init__.py.
    },
    scripts=[],
    install_requires=[
        'absl-py',
        'cached_property',
        'protobuf==3.19.4',
        'google-api-core==2.8.2',
        # TODO(adarob): Replace with 'clu' once >0.0.6 is released.
        'clu==0.0.8',
        'flax==0.6.3',
        'gin-config',
        f'jax==0.3.25',
        f'jaxlib==0.3.25',
        'numpy',
        'orbax==0.0.2',
        't5==0.9.4',
        'tensorflow==2.11.1',
        'einops',
        'tfds-nightly==4.8.3.dev202304050043',
        'tensorflow_probability==0.19.0',
        'tensorflow-addons==0.19.0',
        'tensorflow-datasets==4.8.3',
        'pycocoevalcap',
        'tensorstore >= 0.1.20',
        'librosa',
        'scikit-image',
        'wandb==0.14.0',
        "optax==0.1.4",
        "tqdm",
        "transforms3d==0.4.1",
        "pyglove==0.4.3",
        "seqio==0.0.8",
    ],
    extras_require={
        'data': ['datasets', 'google-cloud-storage', "resampy"],
        "demo": ["resampy", 'google-cloud-storage', 'gradio==4.8.0', 'notebook', 'sk-video'],
        # Cloud TPU requirements.
        'tpu': [f'jax[tpu]==0.3.25'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)