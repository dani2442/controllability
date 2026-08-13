"""Run every paper experiment and regenerate all numerical artifacts."""

from experiments.exp01_discretization import run as run01
from experiments.exp02_controllability import run as run02
from experiments.exp03_lqr import run as run03
from experiments.exp04_conditioning import run as run04
from experiments.common import parser


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    for run in (run01, run02, run03, run04):
        result = run(args.quality)
        written = [result["figure"].name]
        if result["table"] is not None:  # exp01 reports its fits in the figure
            written.append(result["table"].name)
        print(f"{run.__module__}: wrote {', '.join(written)}")
