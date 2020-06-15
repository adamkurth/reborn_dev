.. _tips:

Tips
====

This is the page for miscellaneous tips that might be helpful when using reborn.

Making your code run faster: profiling with ipython
---------------------------------------------------

If you want to write faster code, you need to identify the parts of the code that take up most of the runtime.
This is called "profiling", and there are many ways to do it.  Here is one way: open an ipython session, and run the
following:

.. code-block::

    %prun -T log.txt exec(open('profile_me.py').read())

Let's look at the output of the above command.  We will use the following ``profile_me.py`` file as an example:

.. code-block:: python

    # Filename: profile_me.py
    import time
    def fast_function():
        time.sleep(0.01)
    def slow_function():
        time.sleep(0.1)
        fast_function()
    def main_function():
        slow_function()
    for i in range(10):
        main_function()

In this example, the functions do nothing other than ``sleep`` for 1 second or 100 ms.
Over the 10 iterations, we expect that ``slow_function`` runs for a cumulative time of 1 second, while
``fast_function`` runs for 100 ms.  The ``main_function`` function should thus run for 1.1 seconds since it contains both
of these functions.  Here is the profile:

.. code-block::

    In [3]: %prun -T log.txt exec(open('profile_me.py').read())
             63 function calls (62 primitive calls) in 1.105 seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
           20    1.103    0.055    1.103    0.055 {built-in method time.sleep}
          2/1    0.000    0.000    1.105    1.105 {built-in method builtins.exec}
            1    0.000    0.000    0.000    0.000 {built-in method io.open}
           10    0.000    0.000    1.104    0.110 <string>:5(slow_function)
           10    0.000    0.000    0.102    0.010 <string>:3(fast_function)
            1    0.000    0.000    1.105    1.105 <string>:1(<module>)
            1    0.000    0.000    1.104    1.104 <string>:2(<module>)
           10    0.000    0.000    1.104    0.110 <string>:8(main_function)
            1    0.000    0.000    0.000    0.000 {method 'read' of '_io.TextIOWrapper' objects}
            1    0.000    0.000    0.000    0.000 _bootlocale.py:33(getpreferredencoding)
            1    0.000    0.000    0.000    0.000 codecs.py:319(decode)
            1    0.000    0.000    0.000    0.000 codecs.py:309(__init__)
            1    0.000    0.000    0.000    0.000 {built-in method _locale.nl_langinfo}
            1    0.000    0.000    0.000    0.000 {built-in method _codecs.utf_8_decode}
            1    0.000    0.000    0.000    0.000 codecs.py:260(__init__)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

    *** Profile printout saved to text file 'log.txt'.

As we can see, the ``sleep`` function runs for a cumulative time of 1.1 second, and so does the ``main_function``
function. Note that ``tottime`` is the total time spent in the function alone (not counting the time spent in nested
functions). ``cumtime`` is the total time spent in the function *plus* all functions called within.  Based on
``tottime`` we conclude that we need to speed up the ``sleep`` function if possible.  Speeding up any other function
is a waste of time because everything else amounts to a fraction of a percent of the total runtime.
