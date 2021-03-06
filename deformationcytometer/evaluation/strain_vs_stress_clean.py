# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in 
# the file 'all_data.txt' located at the same directory as this script 
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deformationcytometer.includes.includes import getInputFile, getConfig, getData
#refetchTimestamps,
from deformationcytometer.evaluation.helper_functions import  getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from deformationcytometer.evaluation.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from deformationcytometer.evaluation.helper_functions import storeEvaluationResults
from deformationcytometer.evaluation.helper_functions import load_all_data

""" loading data """
# get the results file (by config parameter or user input dialog)
datafile = getInputFile(filetype=[("txt file", '*_result.txt')])

# load the data and the config
data, config = load_all_data(datafile)

fitStiffness(data, config)

""" plotting data """

initPlotSettings()

# add multipage plotting
pp = PdfPages(datafile[:-11] + '.pdf')

# generate the velocity profile plot
plotVelocityProfile(data, config)
pp.savefig()
plt.cla()

# generate the stress strain plot
plotStressStrain(data, config)
pp.savefig()

# generate the info page with the data
plotMessurementStatus(data, config)

pp.savefig()
#plt.show()
pp.close()

# store the evaluation data in a file
storeEvaluationResults(data, config)