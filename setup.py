from setuptools import setup

# For more information about uploading the python package to PyPI, please check the link:
# https://github.com/judy2k/publishing_python_packages_talk

# Use README.md as the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


if __name__ == '__main__':
    setup(
        name="mdr-library",
        use_scm_version=True,
        author="Kanishka Sharma, Joao Almeida e Sousa and Steven Sourbron",
        author_email="kanishka.sharma@sheffield.ac.uk, j.g.sousa@sheffield.ac.uk, s.sourbron@sheffield.ac.uk",
        description="Open-source, platform independent library for Model Driven Registration (MDR) in quantitative renal MRI",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/QIB-Sheffield/MDR-Library",
        license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
        python_requires='>=3.6, <4',
        packages=['MDR'],
        install_requires=["numpy", "pandas", "SimpleITK", "itk-elastix"],
        setup_requires=['setuptools_scm'],
        include_package_data=True,
        keywords=['python', "medical imaging", "DICOM", "MRI", "renal", "kidney", "motion correction", "registration"],
        # Classifiers - the purpose is to create a wheel and upload it to PYPI
        classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Environment :: Console',
            'Operating System :: OS Independent',

            'Programming Language :: Python :: 3',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate you support Python 3. These classifiers are *not*
            # checked by 'pip install'. See instead 'python_requires' below.
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',

            # Pick your license as you wish
            'License :: OSI Approved :: Apache Software License',
        ],
    )
