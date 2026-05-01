from core.ensemble import generate_structured_ensemble
from analysis.frequency import frequency_analysis
from analysis.cooccurrence import pair_analysis
from analysis.stability import split_blocks, stability_index

def pipeline_v2(
    n,
    target_h,
    target_delta,
    h_last
):
    ensemble = generate_structured_ensemble(
        n,
        target_h,
        target_delta,
        h_last
    )

    # solo sestine
    sestine = [x[0] for x in ensemble]

    freq = frequency_analysis(sestine)
    pairs = pair_analysis(sestine)

    # blocchi
    blocks = split_blocks(ensemble)
    stability = stability_index(blocks)

    return {
        "ensemble": ensemble,
        "freq": freq,
        "pairs": pairs,
        "stability": stability
    }
