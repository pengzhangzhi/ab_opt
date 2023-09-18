import ray
from src.tools.eval.run import main, parse_args

if __name__ == '__main__':
    ray.init()
    args = parse_args()
    main(args)
