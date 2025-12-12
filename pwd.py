# Minimal stub for Unix-only `pwd` module so imports don't crash on Windows.

def getpwuid(uid):
    # This should never actually be called in your current project.
    # If it is, you'll see this error.
    raise NotImplementedError("pwd module is not available on Windows.")
