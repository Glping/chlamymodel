import FMM.runSperm as runSperm
import extern.FMBEM as FMBEM
import FMM.handlerecipes as handlerecipes

import numpy as np


sperms = {'sperm': runSperm.buildMovingSpermRecipe(
    0,
    translation=(0, 0, 0),
    rotation=np.diag([1, 1, 1]),
    frequency=50,
    length=50,
    spatialResolution=30,
    headGrid=40,
    timeResolution=100,
    mean_bending_ratio=1)}

FMBEM.runit(handlerecipes.transformed_shape(sperms,
    velocity=(1, 1, 1)), ['sperm'], 'example_sperm_swimming', 'sperm_test')
