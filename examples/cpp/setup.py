from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='for_sens_vdp',  # the name of the module
    ext_modules=[
        CppExtension(
            name='for_sens_vdp_act_fn',               # the module name in Python
            sources=['for_sens_vdp_act_fn.cpp'],      # our C++ file
            include_dirs=[                            #Change to sundials path
                '/opt/sundials-7.2.1/include/',
            ],  
            extra_link_args=[                         #Change to sundials library path
                '-L/opt/sundials-7.2.1/lib',
                '-lsundials_cvodes', 
                '-lsundials_nvecserial', 
                '-lsundials_sunmatrixdense', 
                '-lsundials_sunlinsoldense', 
                '-lsundials_core'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)