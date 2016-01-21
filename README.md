The Recursive Hessian Sketch for Adaptive Filtering
===================================================

This is the companion code that was used to produce the figures
of the paper __The Recursive Hessian Sketch for Adaptive Filtering__
by Robin Scheibler and Martin Vetterli, submitted to ICASSP 2016.

Authors
-------

Robin Scheibler, and Martin Vetterli are with 
Laboratory for Audiovisual Communications ([LCAV](http://lcav.epfl.ch)) at 
[EPFL](http://www.epfl.ch).

<img src="http://lcav.epfl.ch/files/content/sites/lcav/files/images/Home/LCAV_anim_200.gif">

#### Contact

[Robin Scheibler](mailto:ivan[dot]dokmanic[at]epfl[dot]ch) <br>
EPFL-IC-LCAV <br>
BC Building <br>
Station 14 <br>
1015 Lausanne

Run the code
------------

All the code is pure python and uses only numpy, scipy, matplotlib. The code was
run with ipython.

    $ ipython --version
    3.2.1

We use anaconda to install python, numpy, matplotlib, etc.

### Code organization

All the classical adaptive filters are implemented in `adaptive_filters.py`.

The proposed algorithm is in `sketch_rls.py`.

### Figures 2. 

Simply run

    $ ipython ./figure_Complexity.py

### Figures 3.

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

License
-------

Copyright (c) 2016, LCAV

This code is free to reuse for non-commercial purpose such as academic or
educational. For any other use, please contact the authors.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" />
</a><br/>
<span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Sketch RLS</span> 
by <a xmlns:cc="http://creativecommons.org/ns#" href="http://lcav.epfl.ch" property="cc:attributionName" rel="cc:attributionURL">LCAV, EPFL</a> 
is licensed under a 
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.<br />
Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/LCAV/sketchrls" rel="dct:source">https://github.com/LCAV/sketchrls</a>.

