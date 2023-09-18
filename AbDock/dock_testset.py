from src.tools.runner.design_for_testset import main, parse_args, eval_all

if __name__ == "__main__":
    args = parse_args()
    if args.eval_all:
        eval_all(args)
    else:
        main(args)
