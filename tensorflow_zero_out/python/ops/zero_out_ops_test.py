from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
from counter_ops import create_counter, increment_counter


class CounterTest(test.TestCase):
  
  def test_increment(self):
    with self.test_session():
      counter = create_counter()
      increment_counter(counter)
      increment_counter(counter)
      increment_counter(counter)
      increment_counter(counter)

# outputs the following:      
'''
count is 0. now it's 1
count is 1. now it's 2
count is 2. now it's 3
count is 3. now it's 4
INFO:tensorflow:time(__main__.CounterTest.test_increment): 0.01s
I0630 13:58:10.023093 140402655590144 test_util.py:2103] time(__main__.CounterTest.test_increment): 0.01s
[       OK ] CounterTest.test_increment
[ RUN      ] CounterTest.test_session
[  SKIPPED ] CounterTest.test_session
[ RUN      ] ZeroOutTest.testZeroOut
'''
