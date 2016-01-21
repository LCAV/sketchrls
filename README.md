The Recursive Hessian Sketch for Adaptive Filtering
===================================================

This is the companion code that was used to produce the figures
of the paper __The Recursive Hessian Sketch for Adaptive Filtering__
by Robin Scheibler and Martin Vetterli, submitted to ICASSP 2016.

At this point in time, the code is intended for the reviewers of the
paper only. All rights are reserved by the authors. Please do note
make this code public.

Once the review process is over and the paper revised, the code will
be made public with a suitable open source license.

Code organization
-----------------

All the classical adaptive filters are implemented in `adaptive_filters.py`.

The proposed algorithm is in `sketch_rls.py`.

Run the code
------------

All the code is pure python and uses only numpy, scipy, matplotlib. The code was
run with ipython.

    $ ipython --version
    3.2.1

We use anaconda to install python, numpy, matplotlib, etc.

Figures 2. 
----------

Simply run

    $ ipython ./figure_Complexity.py

Figures 3.
----------

Start an ipython cluster in the repository.

    $ ipcluster start -n x

where `x` is the number of engines you want to use. You can change the number
of loops directly in the script line 42. Then, run the command

    $ ipython figure_MSE_sim.py

This will run the long simulation needed. The result will be stored
in the folder `sim_data` and the name of the file will contain the date and time.

Copy the date and time in the file `figure_MSE_plot.py` line 61-64. Then run

    $ ipython figure_MSE_plot.py

Finally, the file `figure_MSE_test.py` allows to be quickly edited to test
different parameters.

    $ ipython figure_MSE_test.py

