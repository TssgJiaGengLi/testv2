import os
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/unixodbc'
print(os.environ['DYLD_LIBRARY_PATH'])