"""Analytical figure for Note S9. Run with --check to verify the supplied CSV."""
from pathlib import Path
from fractions import Fraction
import argparse
import csv
import math


def horizon_curves(n=288):
    r = Fraction(201, 200)
    k = r**2 - 1
    rho, beta = 2, Fraction(1, 20)
    kappa = k / (rho * r - 1)
    regional_floor = 1 - kappa
    system_floor = regional_floor - beta
    assert 0 < system_floor < regional_floor < 1
    allowance = total_loss = 0.0
    rows = []
    first_saturated = None
    for issue in range(1, n+1):
        displacement = min(1, math.sqrt(float(k) * allowance) / float(r))
        if displacement == 1 and first_saturated is None:
            first_saturated = issue
        residual = 1 - displacement
        analytical_residual = max(0, (1+r)*r**(1-issue)-r)
        assert math.isclose(residual, float(analytical_residual), abs_tol=1e-14)
        total_loss += residual**2
        allowance += float(r**2) - residual**2
        rows.append((issue, math.sqrt(total_loss/issue),
                     float(regional_floor), float(system_floor)))
    assert first_saturated == 140
    assert round(rows[71][1], 5) == 0.70569
    assert round(rows[143][1], 5) == 0.52118
    return rows


def render(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8.2, 'pdf.fonttype': 42,
                         'font.family': 'DejaVu Sans'})
    horizons, exact, region, system = zip(*rows)
    fig, ax = plt.subplots(figsize=(5.75, 2.25), layout="constrained")
    ax.plot(horizons, exact, color='black', lw=1.65,
            label='Full projection (region and total)')
    ax.plot(horizons, region, color='.30', dashes=(6, 2), lw=1.35,
            label='Fixed group: regional lower bound')
    ax.plot(horizons, system, color='.32', dashes=(1.5, 2), lw=1.35,
            label='Fixed group: system lower bound')
    for t in (72, 144):
        value = exact[t-1]
        ax.plot(t, value, 'o', ms=3.4, color='black')
        ax.annotate(f'{value:.3f}', (t, value), xytext=(4, 4),
                    textcoords='offset points', fontsize=8)
    ax.set(xlim=(1, 288), ylim=(0, 1.07), xticks=[1, 72, 144, 216, 288],
           ylabel=r'RMSE / $D$', xlabel=r'Number of releases $T$')
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='y', color='.88', lw=.5)
    ax.legend(loc='lower left', frameon=True, framealpha=1, edgecolor='none', fontsize=7.2)
    fig.savefig(output, metadata={'Title': 'Horizon dependence in the fixed-partition construction'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    dest = Path(__file__).resolve().parent
    rows = horizon_curves()
    fields = ('T','full_exact_region_and_system_over_D','group_lower_bound_region_over_D',
              'group_lower_bound_system_over_D')
    path = dest/'group_horizon.csv'
    if args.check:
        with path.open(encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            assert tuple(next(reader)) == fields
            archived = [tuple(map(float, row)) for row in reader]
        assert archived == rows
        print(f'All {len(rows)} horizons match the recurrence and closed form.')
    else:
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)
        render(rows, dest/'Fig_S9_Group_Horizon.pdf')
