"""Run every paper experiment and regenerate all numerical artifacts."""

from experiments.common import parser
from experiments.conditioning import run as run_conditioning
from experiments.controllability import run as run_controllability
from experiments.lqr import run as run_lqr


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    for run in (run_controllability, run_lqr, run_conditioning):
        result = run(args.quality)
        print(f"{run.__module__}: wrote {result['figure'].name}, {result['table'].name}")
