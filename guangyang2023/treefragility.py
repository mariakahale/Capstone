def tree_damage_probability(ice_thickness_cm: float) -> float:
    """
    Per-tree probability of ice-storm damage to a hazard tree.
    
    From Hou et al. (2023), based on satellite-derived tree cover 
    change during the Oct 2020 Oklahoma ice storm.
    
    Parameters
    ----------
    ice_thickness_cm : float
        Radial ice thickness in cm. Valid range: 0.25 - 3.8 cm.
        Values outside this range are extrapolations the paper
        did not validate.

    Returns
    -------
    float
        Probability (0-1) that a single hazard tree sustains 
        damage sufficient to potentially fall on the line.

        tree_damage_probability(ice) answers: "If I pick one specific tree near the line, what's the chance THIS tree gets damaged badly enough to be a threat?"
    """
    if not (0.25 <= ice_thickness_cm <= 3.8):
        # Not a hard error, but flag it — outside the fitted range
        print(f"Warning: ice_thickness_cm={ice_thickness_cm} is outside "
              f"the paper's validated range [0.25, 3.8] cm")
    
    p_single_tree = 0.125 + 0.101 * ice_thickness_cm
    return min(max(p_single_tree, 0.0), 1.0)  # clip to valid probability


def segment_tree_failure_probability(ice_thickness_cm: float, 
                                       n_hazard_trees: int) -> float:
    """
    Probability that at least one hazard tree along a line segment
    fails and threatens the line, given n independent hazard trees.
    
    This is Eq. (3) from Hou et al. (2023):
        p_tree,j = 1 - (1 - p_tree,iw)^n
    
    Parameters
    ----------
    ice_thickness_cm : float
        Radial ice thickness in cm.
    n_hazard_trees : int
        Number of hazard trees identified along this line segment
        (from GIS/vegetation survey — a separate input, not derived
        from ice thickness).

    Returns
    -------
    float
        Probability (0-1) that at least one hazard tree along the
        segment fails.
    segment_tree_failure_probability(ice, n) answers a different question: "This line segment has n of these trees standing near it. What's the chance that AT LEAST ONE of them falls and hits the line?"
    """
    p_single_tree = tree_damage_probability(ice_thickness_cm)
    p_segment = 1 - (1 - p_single_tree) ** n_hazard_trees
    return p_segment


# Example usage
if __name__ == "__main__":
    ice = 2.5  # cm
    
    print(f"Per-tree failure probability at {ice} cm ice: "
          f"{tree_damage_probability(ice):.3f}")
    
    for n in [0, 1, 3, 5, 10]:
        p = segment_tree_failure_probability(ice, n)
        print(f"  n={n} hazard trees -> segment risk = {p:.3f}")