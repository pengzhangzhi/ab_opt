import ray
from src.tools.relax.run import run_relax, main, parse_args


if __name__ == '__main__':
    ray.init()
    args = parse_args()
    main(args)