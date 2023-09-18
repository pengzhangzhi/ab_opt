from src.tools.runner.design_for_pdb import args_from_cmdline, dock_for_pdb

if __name__ == '__main__':
    dock_for_pdb(args_from_cmdline())
