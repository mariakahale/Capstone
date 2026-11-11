File	                    Role
wire_fragility_by_span.py	The wire model: how much ice and wind a wire of a given length takes before it snaps. Not run by the pipeline; the build script uses it.

build_wire_fig7d_lookup.py	Run once. Uses the wire model to compute failure probability for every span, ice thickness and wind speed, and saves → wire_fig7d_surfaces_by_span.npz

build_pole_fig7_lookup.py	Same idea for poles (every span, age and phase count) → pole_fig7_surfaces_by_span.npz

wire_fig7d_lookup.py	What the pipeline calls for wires. Give it span, ice and wind, and it reads the .npz and returns the probability.

pole_fig7_lookup.py	What the pipeline calls for poles. Give it age, tributary span, phases, ice and wind.

the two .npz files	The saved tables the lookups read

plot_fig7_surfaces.py	Draws the 3D surfaces from the .npz files, so you can see them → fig7_surface_plots/

NOTE_wire_stiffness_calibration.txt	Your open note on the wire stiffness (EA) calibration: the 117.3 N/m match from before

the PDF	Short (2005), the textbook the conductor specs come from
