from setuptools import setup, find_packages, Command

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys
        import subprocess 
        errno = subprocess.call(['py.test', 'tests'])
        raise SystemExit(errno)

setup( name='genomic_neuralnet'
     , version='0.0.1'
     , description='Evaluation of various genomic selection and whole genome prediction methods on public datasets'
     , author='Riley McDowell'
     , author_email='mcdori02_at_luther_dot_edu'
     , url='https://github.com/rileymcdowell/genomic-neuralnet'
     , packages=find_packages()
     , cmdclass = { 'test': PyTest }
     )


