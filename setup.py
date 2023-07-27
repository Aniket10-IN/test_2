from setuptools import find_packages, setup
from typing import List

hyphen_edot = '-e .'
def get_requirements(filepath:str)-> List[str]:
    requirements = []

    with open (filepath) as obj:
        requirements =   obj.readlines()
        requirements = [req.replace('\n', ' ')for req in requirements]
        # requirements = [req for req in requirements if req != '-c .']
        if hyphen_edot in requirements:
            requirements.remove(hyphen_edot)

    return requirements


setup(
    name =  'rbm_cb_Project',
    version = '0.0.1',
    author = 'Aniket',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)