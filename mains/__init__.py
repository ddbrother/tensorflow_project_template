
import sys
import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir)  
print(__file__, ": add project's root path", parentdir)
