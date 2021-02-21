from distutils.core import setup, Command
from distutils.extension import Extension
from Cython.Distutils import build_ext

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)

setup(
    cmdclass = {'build_ext': build_ext, 'test': PyTest},
    ext_modules = [Extension("optspace",
                             sources=["OptSpace_C/las2.c",
                                      "OptSpace_C/matops.c",
                                      "OptSpace_C/rand.c",
                                      "OptSpace_C/svdlib.c",
                                      "OptSpace_C/svdutil.c",
                                      "OptSpace_C/OptSpace.c",
                                      "optspace.pyx"],
                             include_dirs=["OptSpace_C"])],
)
