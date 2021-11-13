from gui import display_GUI
from aux import model, predict
import glob
import argparse

if __name__ == '__main__':
    """
    1. Set step_parameter of the step you want to run.
    2. Run this python script with the step's name as argument.
    """

    parser  = argparse.ArgumentParser()
    subparsers              = parser.add_subparsers(help='select a step', dest="step")
    parser_fitmodel            = subparsers.add_parser('model',                            help='fit regressors with data as csv')
    parser_predictwithgui      = subparsers.add_parser('predict',                          help='produces a gui to predict patients disease')
    
    parser_fitmodel.add_argument('-i','--input-csv',     dest='inpath',    type=str, help='path to csv file for fitting.',  required=True)
    parser_fitmodel.add_argument('-o','--output-file',     dest='outpath',    type=str, help='path to output for model.',  required=True)

    #parser_predictwithgui.add_argument('-m','--model-path',     dest='model_path',    type=str, help='path to for model file',  required=True)
    
    args = parser.parse_args()

    if args.step == 'model':
        inpath  = args.inpath
        outpath = args.outpath
        model_ = model(inpath, outpath)

    if args.step == 'predict':
        display = display_GUI()
        display()