.. Tomosynthesis Machine Learning documentation master file, created by
   sphinx-quickstart on Sat Dec 06 14:25:42 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Tomosynthesis Machine Learning's documentation!
==========================================================

Requirements
------------
* `PIL <http://www.pythonware.com/products/pil/>`_
* `Numpy <http://www.numpy.org/>`_
* `scikit_learn <http://scikit-learn.org/stable/>`_
* `scipy <http://www.scipy.org/>`_
* `skimage <http://scikit-image.org/docs/dev/api/skimage.html>`_
* `Matplotlib 1.3 <http://www.matplotlib.org>`_  (optional for plotting)
The main packages required are listed above, however, there are other minor packages may required in modeuls, please install accordingly.
	
Contents
--------

This documentation includes six sections.Test examples are given within module comments for simple utility functions, script for running tasks are given in the last section. 

* Core Functions: includes basic image input/output modules, image class modeule, patch class mdule, calcification class module.

* Preprocessing: includes denoising module, contrast enhancement module and background substraction module.

* Mass Detection: includes initial detection module, segmentation module, feature extraction module and classification module.

* Micro Calcification Detection: includes foreground substraction, LOG filtering and post constraining.

* Bilateral Asymmetric Analysis: includes boundary detection, fiducial point selection, registration and region comparison.

* Test Examples: includes test examples for how to run each of the above task.



.. toctree::
   :numbered:
   
   core_functions
   preprocessing
   mass_detection
   micro_calcification_detect
   asymmetric_analysis
   test_examples
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

