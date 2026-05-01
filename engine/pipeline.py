from engine.run_simulation import run_simulation
from analysis.frequency import frequency_analysis
from analysis.cooccurrence import pair_analysis

def pipeline(n=100000):
    ensemble = run_simulation(n)

    only_sestine = [x[0] for x in ensemble]

    freq = frequency_analysis(only_sestine)
    pairs = pair_analysis(only_sestine)

    return {
        "freq": freq,
        "pairs": pairs
    }
