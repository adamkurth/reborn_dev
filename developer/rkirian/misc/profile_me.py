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
